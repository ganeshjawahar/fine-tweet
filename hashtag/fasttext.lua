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
cmd:option('-data', '../../../data/auxillary/hashtag/', '')
cmd:option('-lr', 0.01, '')
cmd:option('-batch_size', 256, '')
cmd:option('-num_epochs', 5, '')
cmd:option('-dropout_p', 0.8, '')
cmd:option('-dim', 10, '')
cmd:option('-rand_data', 0, '')
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
    if params.rand_data == 0 then return words end
    local rw, idx = {}, torch.randperm(#words)
    for i = 1, #words do
    	table.insert(rw, words[idx[i]])
    end
    return rw
end

params.word2index, params.index2word = tds.hash(), tds.hash()
function get_neg(tweet, tag)
	local ug = stringx.split(tweet, ' ')
	local neg_cands = {}
	for i = 1, #ug do
		if ug[i] ~= tag then
			table.insert(neg_cands, ug[i])
		end
	end
	if #neg_cands == 0 then return nil end
	return neg_cands[math.random(#neg_cands)]
end
params.max_len = -1
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
function get_tensors(data_file, add_2_word_map)
	local tensors = {}
	for line in io.lines(data_file) do
		local dcontent = stringx.split(line, '\t')			
		if #dcontent[1] > 0 then
			local htag = dcontent[2]
			local neg_tag = get_neg(dcontent[1], dcontent[2])
			if neg_tag ~= nil then
				if add_2_word_map ~= nil then
					if params.cand2index[htag] == nil then
						params.cand2index[htag] = #params.index2cand + 1
						params.index2cand[#params.index2cand + 1] = htag
					end
					if params.cand2index[neg_tag] == nil then
						params.cand2index[neg_tag] = #params.index2cand + 1
						params.index2cand[#params.index2cand + 1] = neg_tag
					end
				end
				if params.cand2index[htag] ~= nil and params.cand2index[neg_tag] ~= nil then
					local words = tokenizeTweet(dcontent[1])
					local input = torch.Tensor(#words)
					for i = 1, #words do
						local word = words[i]
						if params.word2index[word] == nil then
							input[i] = params.word2index[params.UK]
						else
							input[i] = params.word2index[word]
						end
					end				
					table.insert(tensors, {input, params.cand2index[htag], 1, #stringx.split(dcontent[1])})
					table.insert(tensors, {input, params.cand2index[neg_tag], 2, #stringx.split(dcontent[1])})
				end
			end
			if #stringx.split(dcontent[1]) > params.max_len then
				params.max_len = #stringx.split(dcontent[1])
			end
		end
		-- if #tensors == 200 then break end
	end
	return tensors
end

params.train_input, params.train_word_count = get_tensors(params.data..'train.tsv', true)
params.dev_input, _ = get_tensors(params.data..'dev.tsv')
params.test_input, _ = get_tensors(params.data..'test.tsv')

params.word_model = nn.Sequential():add(nn.Sequencer(nn.Sequential():add(nn.LookupTable(#params.index2word, params.dim)):add(nn.Mean()))):add(nn.JoinTable(1)):add(nn.View(-1, params.dim))
params.cand1_model = nn.Sequential():add(nn.LookupTable(#params.index2cand, params.dim)):add(nn.Squeeze())
params.model = nn.Sequential()
params.model:add(nn.ParallelTable())
params.model.modules[1]:add(params.word_model)
params.model.modules[1]:add(params.cand1_model)
params.model:add(nn.JoinTable(2))
-- params.model:add(nn.Linear(params.dim * 2, params.dim * 2))
params.model:add(nn.ReLU())
params.model:add(nn.Linear(params.dim * 2, 2))
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
		local cur_input_sent, cur_input_word1, cur_target = {}, torch.CudaTensor(cur_bsize, 1), torch.CudaTensor(cur_bsize, 1)
		for j = 1, cur_bsize do
			local record = params.train_input[indices[i + j - 1]]			
			table.insert(cur_input_sent, record[1]:cuda())
			cur_input_word1[j][1] = record[2]
			cur_target[j][1] = record[3]
		end
		local feval = function(x)
			params.pp:copy(x)
			params.gp:zero()
			local out = params.model:forward({cur_input_sent, cur_input_word1})
			local loss = params.criterion:forward(out, cur_target)
			epoch_loss = epoch_loss + loss * cur_bsize
			local grads = params.criterion:backward(out, cur_target)
			params.model:backward({cur_input_sent, cur_input_word1}, grads)
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
	for i = 1, 2 do
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
		local cur_input_sent, cur_input_word1, cur_target = {}, torch.CudaTensor(cur_bsize, 1),  {}
		for j = 1, cur_bsize do
			local record = input[i + j - 1]
			table.insert(cur_input_sent, record[1]:cuda())
			cur_input_word1[j][1] = record[2]
			table.insert(cur_target, record[3])
		end
		local out = params.model:forward({cur_input_sent, cur_input_word1})
		local soft = params.soft_max:forward(out)
		_, ids = soft:max(2)
		for j = 1, cur_bsize do
			local ori_len = input[i + j - 1][4]
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
	for i = 1, 2 do
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