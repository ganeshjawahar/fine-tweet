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
tds = require('tds')

cmd = torch.CmdLine()
cmd:option('-data', '../../../data/auxillary/rel_capt/', '')
cmd:option('-rep', '../../auxillary/representations/rel_capt/bow/', '')
cmd:option('-hid_size', 100, '')
cmd:option('-lr', 0.01, '')
cmd:option('-batch_size', 256, '')
cmd:option('-num_epochs', 5, '')
cmd:option('-dropout_p', 0.8, '')
params = cmd:parse(arg)
torch.manualSeed(123)

params.word2index, params.index2word = {}, {}
function get_tags(tweet)
	local ug = stringx.split(tweet, ' ')
	local cap_words = {}
	for i = 1, #ug do
		if stringx.isupper(ug[i]:sub(1,1)) == true then
			table.insert(cap_words, ug[i])
		end
	end
	return cap_words
end
params.max_len = -1
function get_tensors(data_file, rep_file, add_2_word_map)
	local dptr, rptr = io.open(data_file, 'r'), io.open(rep_file, 'r')
	local tensors = {}
	while true do
		local dline, rline = dptr:read(), rptr:read()
		if dline == nil or rline == nil then
			break
		end
		local dcontent = stringx.split(dline, '\t')			
		local rcontent = stringx.split(rline, ',')
		if #dcontent[1] > 0 and #rcontent > 1 then
			local label = dcontent[2]
			local tags = get_tags(dcontent[1])
			if #tags > 0 then
				for a = 1, #tags do
					local tag = tags[a]
					if add_2_word_map ~= nil then
						if params.word2index[tag] == nil then
							params.word2index[tag] = #params.index2word + 1
							params.index2word[#params.index2word + 1] = tag
						end
					end
					if params.word2index[tag] ~= nil  then
						local input = {}
						for i = 1, #rcontent - 1 do
							local temp = stringx.split(rcontent[i],":")				
							table.insert(input, {tonumber(temp[1]), tonumber(temp[2])} )
						end
						table.insert(tensors, {input, params.word2index[tag], tonumber(label) + 1, #stringx.split(dcontent[1])})
					end
				end
			end
			if #stringx.split(dcontent[1]) > params.max_len then
				params.max_len = #stringx.split(dcontent[1])
			end
		end
		-- if #tensors == 200 then break end
	end
	io.close(dptr)
	io.close(rptr)
	return tensors
end

params.train_input, params.train_word_count = get_tensors(params.data..'train.tsv', params.rep..'train.tsv', true)
params.test_input, _ = get_tensors(params.data..'test.tsv', params.rep..'test.tsv')
print('#candidates = '..#params.index2word)

params.cand_model = nn.Sequential():add(nn.LookupTable(#params.index2word, 200)):add(nn.Squeeze())
params.model = nn.Sequential()
params.model:add(nn.ParallelTable())
params.model.modules[1]:add(nn.SparseLinear(50000, params.hid_size))
params.model.modules[1]:add(params.cand_model)
params.model:add(nn.JoinTable(2))
params.model:add(nn.ReLU())
params.model:add(nn.Linear(params.hid_size + 200, 2))
params.model:add(nn.Dropout(params.dropout_p))
params.model = params.model:cuda()
params.criterion = nn.CrossEntropyCriterion():cuda()
params.optim_state = {learningRate = params.lr}
params.pp, params.gp = params.model:getParameters()

print('training...')
function train(states)
	params.model:training()
	local indices = torch.randperm(#params.train_input)
	local epoch_loss = 0
	xlua.progress(1, #params.train_input)
	for i = 1, #params.train_input, params.batch_size do
		local cur_bsize = math.min(i + params.batch_size - 1, #params.train_input) - i + 1
		local cur_input_sent, cur_input_word, cur_target = {}, torch.CudaTensor(cur_bsize, 1), torch.CudaTensor(cur_bsize, 1)
		for j = 1, cur_bsize do
			local record = params.train_input[indices[i + j - 1]]			
			table.insert(cur_input_sent, torch.CudaTensor(record[1]))
			cur_input_word[j][1] = record[2]
			cur_target[j][1] = record[3]
		end
		local feval = function(x)
			params.pp:copy(x)
			params.gp:zero()
			local out = params.model:forward({cur_input_sent, cur_input_word})
			local loss = params.criterion:forward(out, cur_target)
			epoch_loss = epoch_loss + loss * cur_bsize
			local grads = params.criterion:backward(out, cur_target)
			params.model:backward({cur_input_sent, cur_input_word}, grads)
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
		local cur_input_sent, cur_input_word, cur_target = {}, torch.CudaTensor(cur_bsize, 1), {}
		for j = 1, cur_bsize do
			local record = input[i + j - 1]
			table.insert(cur_input_sent, torch.CudaTensor(record[1]))
			cur_input_word[j][1] = record[2]
			table.insert(cur_target, record[3])
		end
		local out = params.model:forward({cur_input_sent, cur_input_word})
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
local test_fscore, res = compute_performance(params.test_input)
print('data='..params.data..';test_score='..test_fscore..';')
print(res)