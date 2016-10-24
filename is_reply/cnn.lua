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
paths.dofile('kim_sent.lua')
tds = require('tds')

cmd = torch.CmdLine()
cmd:option('-data', '../../../data/auxillary/is_reply/', '')
cmd:option('-glove_dir', '/home/ganesh/github/tweet-thought/data/glove/', 'Directory for accesssing the pre-trained glove word embeddings')
cmd:option('-dim', 200, '')
cmd:option('-rand_data', 0, '')
cmd:option('-lr', 0.01, '')
cmd:option('-batch_size', 256, '')
cmd:option('-num_epochs', 5, '')
cmd:option('-dropout_p', 0.8, '')
params = cmd:parse(arg)
torch.manualSeed(123)
params.ZERO = '<zero_cnn>'

function tokenizeTweet(tweet, max_words)
	if max_words == nil then max_words = 1000 end
    local words = {}
    for i = 1, 4 do table.insert(words, params.ZERO) end
    table.insert(words, '<sot>')
    local _words = stringx.split(tweet)
    for i = 1, #_words do
        if #words < max_words then table.insert(words, _words[i]) end
    end
    if #words < max_words then table.insert(words, '<eot>') end
    for i = 1, 4 do if #words < max_words then table.insert(words, params.ZERO) end end
    if params.rand_data == 0 then return words end
    local rw, idx = {}, torch.randperm(#words)
    for i = 1, #words do
    	table.insert(rw, words[idx[i]])
    end
    return rw
end
params.word2index, params.index2word = tds.hash(), tds.hash()
function build_vocab(data_file)
	params.max_words = 0
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
			if #words > params.max_words then
				params.max_words = #words
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
			local words = tokenizeTweet(dcontent[1], params.max_words)
			local tensor = torch.Tensor(params.max_words):fill(params.word2index[params.ZERO])
			for i = 1, #words do
				local word = words[i]
				if params.word2index[word] == nil then
					tensor[i] = params.word2index[params.UK]
				else
					tensor[i] = params.word2index[word]
				end
			end
			if add_2_word_map == true then
				if params.cand2index[dcontent[2]] == nil then
					params.index2cand[#params.index2cand + 1] = dcontent[2]
					params.cand2index[dcontent[2]] = #params.index2cand
				end
			end
			local ug_count = #stringx.split(dcontent[1], ' ')
			if params.cand2index[dcontent[2]] ~= nil then
				table.insert(tensors, {tensor, params.cand2index[dcontent[2]],ug_count})
			end
			if ug_count > params.max_len then
				params.max_len = ug_count
			end
		end
		-- if #tensors == 200 then break end
	end
	return tensors
end

function get_layer(model, name)
	for _, node in ipairs(model.forwardnodes) do
	    if node.data.annotations.name == name then
	        return node.data.module
	    end
	end
	return nil
end

params.train_input, params.train_word_count = get_tensors(params.data..'train.tsv', true)
params.dev_input, _ = get_tensors(params.data..'dev.tsv')
params.test_input, _ = get_tensors(params.data..'test.tsv')
print('classes = '..#params.index2cand)

config = {}
config.ngram_lookup_rows = #params.index2word
config.ngram_lookup_cols = params.dim
config.kernels = '3,4,5'
config.num_feat_maps = 100
config.dropout_p = 0.5
config.max_words = params.max_words
params.cnn_layer, params.out_size = get_model(config)
params.ngram_lookup = get_layer(params.cnn_layer, 'ngram_lookup')

print('initializing the pre-trained embeddings...')
local glove_complete_path = params.glove_dir .. 'glove.twitter.27B.'.. params.dim .. 'd.txt.gz'
local is_present = lfs.attributes(glove_complete_path) or -1
if is_present ~= -1 then
	local start_time = sys.clock()
	local ic = 0
	local start = 0
	local ic = 0
	for line in io.lines(glove_complete_path) do
		local content = stringx.split(line)
		local word = content[1]
		if params.word2index[word] ~= nil then
			local tensor = torch.Tensor(#content - 1)
			for i = 2, #content do
				tensor[i - 1] = tonumber(content[i])
			end
			params.ngram_lookup.weight[start + params.word2index[word]] = tensor
			ic = ic + 1
		end
	end
	print(string.format("%d out of %d words initialized.", ic, #params.index2word))
end

params.model = nn.Sequential()
params.model:add(params.cnn_layer)
params.model:add(nn.ReLU())
params.model:add(nn.Linear(params.out_size, #params.index2cand))
params.model:add(nn.Dropout(params.dropout_p))
params.model = params.model:cuda()
params.criterion = nn.CrossEntropyCriterion():cuda()
params.pp, params.gp = params.model:getParameters()
params.optim_state = {learningRate = params.lr}
params.linear_pred_layer = params.model.modules[3]
params.linear_pred_layer.weight:normal():mul(0.01)
params.linear_pred_layer.bias:zero()

print('training...')
function train(states)
	params.model:training()
	local indices = torch.randperm(#params.train_input)
	local epoch_loss = 0
	xlua.progress(1, #params.train_input)
	for i = 1, #params.train_input, params.batch_size do
		local cur_bsize = math.min(i + params.batch_size - 1, #params.train_input) - i + 1
		local cur_input_sent, cur_target = torch.CudaTensor(cur_bsize, params.max_words), torch.CudaTensor(cur_bsize, 1)
		for j = 1, cur_bsize do
			local record = params.train_input[indices[i + j - 1]]			
			cur_input_sent[j] = record[1]:cuda()
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
		params.ngram_lookup.weight[params.word2index[params.ZERO]]:zero()
		--[[
		-- Renorm (Euclidean projection to L2 ball)
    	local renorm = function(row)
    		local n = row:norm()
      		row:mul(3):div(1e-7 + n)
    	end
	    -- renormalize linear row weights
    	local w = params.linear_pred_layer.weight
    	for j = 1, w:size(1) do
      		renorm(w[j])
    	end
    	]]--
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
		local cur_input_sent, cur_target = torch.CudaTensor(cur_bsize, params.max_words), {}
		for j = 1, cur_bsize do
			local record = input[i + j - 1]
			cur_input_sent[j] = record[1]:cuda()
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
print('data='..params.data..';dev_score='..dev_fscore..';test_score='..test_fscore..';dim='..params.dim..';rand='..params.rand_data..';')
print(res)