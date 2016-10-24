require 'torch'
require 'io'
require 'nn'
require 'sys'
require 'os'
require 'xlua'
require 'lfs'
require 'cunn'
require 'cutorch'
require 'pl.stringx'
require 'pl.file'
require 'optim'
require 'rnn'
tds = require('tds')

cmd = torch.CmdLine()
cmd:option('-data', '../../../../data/auxillary/length/sentiment_', '')
cmd:option('-lr', 0.01, '')
cmd:option('-batch_size', 256, '')
cmd:option('-num_epochs', 5, '')
cmd:option('-dropout_p', 0.8, '')
cmd:option('-dim', 10, '')
cmd:option('-wordNGrams', 2, 'number of n-grams to consider')
params = cmd:parse(arg)
torch.manualSeed(123)

function tokenizeTweet(tweet)
    local words = {}
    local _words = stringx.split(tweet)
    for i = 1, #_words do
    	table.insert(words, _words[i])
    end
    for size = 2, params.wordNGrams do
    	if #_words >= size then
	    	for start = 1, (#_words - size + 1) do
	    		local ngram = ''
	    		for i = 1, size do
	    			ngram = ngram .. _words[start + i - 1] .. '$'
	    		end
	    		table.insert(words, ngram)
	    	end
	    end
    end
    return words
end

params.word2index, params.index2word = tds.hash(), tds.hash()
function build_vocab(data_file)
	for line in io.lines(data_file) do
		local dcontent = stringx.split(line, '\t')			
		if #dcontent[1] > 0 then
			local words = tokenizeTweet(dcontent[1])
			for j = 1, #words do
				local word = words[j]
				if params.word2index[word] == nil then
					params.index2word[#params.index2word + 1] = word
					params.word2index[word] = #params.index2word
				end
			end
		end
	end
	params.UK = '<UK>'
	params.index2word[#params.index2word + 1] = params.UK
	params.word2index[params.UK] = #params.index2word
end
build_vocab(params.data..'train.tsv')
print(#params.index2word .. ' found.')

params.cand2index, params.index2cand = tds.hash(), tds.hash()
params.max_len = -1
function get_tensors(data_file, add_2_word_map)
	local tensors = {}
	for line in io.lines(data_file) do
		local dcontent = stringx.split(line, '\t')			
		if #dcontent[1] > 0 then
			local words = tokenizeTweet(dcontent[1])
			local tensor = torch.Tensor(#words)
			for i = 1, #words do
				local word = words[i]
				if params.word2index[word] == nil then
					tensor[i] = params.word2index[params.UK]
				else
					tensor[i] = params.word2index[word]
				end
			end
			local cur_len = #stringx.split(dcontent[1], ' ')
			local label = dcontent[2]
			if add_2_word_map == true then
				if params.cand2index[label] == nil then
					params.index2cand[#params.index2cand + 1] = label
					params.cand2index[label] = #params.index2cand
				end
			end
			if params.cand2index[label] ~= nil then
				table.insert(tensors, {tensor, params.cand2index[label], cur_len})
			end
			if cur_len > params.max_len then
				params.max_len = cur_len
			end
		end
		-- if #tensors == 200 then break end
	end
	return tensors
end

params.train_input, params.train_word_count = get_tensors(params.data..'train.tsv', true)
params.dev_input, _ = get_tensors(params.data..'dev.tsv')
params.test_input, _ = get_tensors(params.data..'test.tsv')
print('classes = '..#params.index2cand)

params.word_model = nn.Sequential():add(nn.Sequencer(nn.Sequential():add(nn.LookupTable(#params.index2word, params.dim)):add(nn.Mean()))):add(nn.JoinTable(1)):add(nn.View(-1, params.dim))
-- params.cand1_model = nn.Sequential():add(nn.LookupTable(#params.index2cand, params.dim)):add(nn.Squeeze())
params.model = nn.Sequential()
-- params.model:add(nn.ParallelTable())
params.model:add(params.word_model)
-- params.model:add(nn.Linear(params.dim, params.dim))
params.model:add(nn.ReLU())
params.model:add(nn.Linear(params.dim, #params.index2cand))
params.model:add(nn.Dropout(params.dropout_p))
params.model = params.model:cuda()
params.criterion = nn.CrossEntropyCriterion():cuda()
params.pp, params.gp = params.model:getParameters()
params.optim_state = {learningRate = params.lr}

print('training...')
function train(states)
	params.model:training()
	local indices = torch.randperm(#params.train_input)
	local epoch_loss = 0
	xlua.progress(1, #params.train_input)
	for i = 1, #params.train_input, params.batch_size do
		local cur_bsize = math.min(i + params.batch_size - 1, #params.train_input) - i + 1
		local cur_input_sent, cur_target = {}, torch.CudaTensor(cur_bsize, 1)
		for j = 1, cur_bsize do
			local record = params.train_input[indices[i + j - 1]]			
			table.insert(cur_input_sent, record[1]:cuda())
			cur_target[j][1] = record[2]
		end
		local feval = function(x)
			params.pp:copy(x)
			params.gp:zero()
			local out = params.model:forward(cur_input_sent)
			local loss = params.criterion:forward(out, cur_target)
			epoch_loss = epoch_loss + loss * cur_bsize
			local grads = params.criterion:backward(out, cur_target)
			params.model:backward(cur_input_sent, grads)
			return loss, params.gp
		end
		optim.adagrad(feval, params.pp, params.optim_state, states)		
		xlua.progress(i, #params.train_input)
	end
	xlua.progress(#params.train_input, #params.train_input)
	return epoch_loss / #params.train_input
end

params.soft_max = nn.SoftMax():cuda()
function compute_performance(input)
	params.model:evaluate()
	local tp, pred_as, gold_as = {}, {}, {}
	for i = 1, #params.index2cand do
		tp[i] = 0
		pred_as[i] = 0
		gold_as[i] = 0
	end
	local acc, tot = {}, {}
	for i = 1, params.max_len do
		acc[i] = 0
		tot[i] = 0
	end
	for i = 1, #input, params.batch_size do
		local cur_bsize = math.min(i + params.batch_size - 1, #input) - i + 1
		local cur_input_sent, cur_target = {}, {}
		for j = 1, cur_bsize do
			local record = input[i + j - 1]
			table.insert(cur_input_sent, record[1]:cuda())
			table.insert(cur_target, record[2])
		end
		local out = params.model:forward(cur_input_sent)
		local soft = params.soft_max:forward(out)
		_, ids = soft:max(2)
		for j = 1, cur_bsize do
			local ori_len = input[i + j - 1][3]
			if ids[j][1] == cur_target[j] then 
				tp[ids[j][1]] = tp[ids[j][1]] + 1 
				acc[ori_len] = acc[ori_len] + 1
			end
			pred_as[ids[j][1]] = pred_as[ids[j][1]] + 1
			gold_as[cur_target[j]] = gold_as[cur_target[j]] + 1
			tot[ori_len] = tot[ori_len] + 1
		end
	end	
	local tp_sum, prec_den, recall_den = 0, 0, 0
	for i = 1, #params.index2cand do
		tp_sum = tp_sum + tp[i]
		prec_den = prec_den + pred_as[i]
		recall_den = recall_den + gold_as[i]	
	end
	local res = ''
	for i = 1, params.max_len do
		if tot[i] ~= 0 then
			res = res .. i .. '=' .. (acc[i] / tot[i]) .. ';'
		else
			res = res .. i .. '=nan;'
		end
	end
	local micro_prec, micro_recall = (tp_sum / prec_den), (tp_sum / recall_den)
	return ((2 * micro_prec * micro_recall) / (micro_prec + micro_recall)), res
end

local states = {}
for epoch = 1, params.num_epochs do
	local loss = train(states)
	print('Epoch ('..epoch..'/'..params.num_epochs..') Loss = '..loss)
end
local dev_fscore, _ = compute_performance(params.dev_input)
local test_fscore, res = compute_performance(params.test_input)
print('data='..params.data..';dev_score='..dev_fscore..';test_score='..test_fscore)
print(res)
