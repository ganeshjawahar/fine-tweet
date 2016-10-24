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
cmd:option('-data', '../../../data/auxillary/is_reply/', '')
cmd:option('-rep', '../../auxillary/representations/is_reply/doc2vec/', '')
cmd:option('-lr', 0.01, '')
cmd:option('-batch_size', 256, '')
cmd:option('-num_epochs', 5, '')
cmd:option('-dropout_p', 0.8, '')
params = cmd:parse(arg)
torch.manualSeed(123)

params.label2index, params.index2label = {}, {}
params.max_len = -1
function get_tensors(data_file, rep_file)
	local dptr, rptr = io.open(data_file, 'r'), io.open(rep_file, 'r')
	local tensors, word_count = {}, 0
	while true do
		local dline, rline = dptr:read(), rptr:read()
		if dline == nil or rline == nil then
			break
		end
		local dcontent = stringx.split(dline, '\t')			
		local rcontent = stringx.split(rline, '\t')
		if #dcontent[1] > 0 and #rcontent > 0 then
			if params.dim == nil then
				params.dim = #rcontent - 1
				print('found dim as '..params.dim)
			end
			local label = dcontent[2]
			if params.label2index[label] == nil then
				params.label2index[label] = #params.index2label + 1
				params.index2label[#params.index2label + 1] = label
			end
			local input = torch.Tensor(params.dim)
			for i = 1, params.dim do
				input[i] = tonumber(rcontent[i])
			end
			local ug_count = #stringx.split(dcontent[1], ' ')
			table.insert(tensors, {input, params.label2index[label], ug_count})
			word_count = word_count + ug_count
			if ug_count > params.max_len then
				params.max_len = ug_count
			end
		end
		-- if #tensors == 200 then break end
	end
	io.close(dptr)
	io.close(rptr)
	return tensors, word_count
end

params.train_input, params.train_word_count = get_tensors(params.data..'train.tsv', params.rep..'train.tsv')
params.dev_input, _ = get_tensors(params.data..'dev.tsv', params.rep..'dev.tsv')
params.test_input, _ = get_tensors(params.data..'test.tsv', params.rep..'test.tsv')

params.model = nn.Sequential()
params.model:add(nn.Linear(params.dim, params.dim))
params.model:add(nn.ReLU())
params.model:add(nn.Linear(params.dim, #params.index2label))
params.model:add(nn.Dropout(params.dropout_p))
params.model = params.model:cuda()
params.criterion = nn.CrossEntropyCriterion():cuda()
params.token_count, params.total_progress = 0, (params.num_epochs * params.train_word_count)
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
		local cur_input, cur_target = torch.CudaTensor(cur_bsize, params.dim), torch.CudaTensor(cur_bsize, 1)
		for j = 1, cur_bsize do
			local record = params.train_input[indices[i + j - 1]]			
			cur_input[j] = record[1]:cuda()
			cur_target[j][1] = record[2]
			params.token_count = params.token_count + record[3]
		end
		local feval = function(x)
			params.pp:copy(x)
			params.gp:zero()
			local out = params.model:forward(cur_input)
			local loss = params.criterion:forward(out, cur_target)
			epoch_loss = epoch_loss + loss * cur_bsize
			--params.model:zeroGradParameters()
			local grads = params.criterion:backward(out, cur_target)
			params.model:backward(cur_input, grads)
			return loss, params.gp
		end
		optim.adagrad(feval, params.pp, params.optim_state, states)		
		--params.model:updateParameters(params.lr * (1.0 - (params.token_count / params.total_progress)))
		xlua.progress(i, #params.train_input)
	end
	xlua.progress(#params.train_input, #params.train_input)
	return epoch_loss / #params.train_input
end

params.soft_max = nn.SoftMax():cuda()
function compute_performance(input)
	params.model:evaluate()
	local tp, pred_as, gold_as = {}, {}, {}
	for i = 1, #params.index2label do
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
		local cur_input, cur_target = torch.CudaTensor(cur_bsize, params.dim), {}
		for j = 1, cur_bsize do
			local record = input[i + j - 1]
			cur_input[j] = record[1]:cuda()
			table.insert(cur_target, record[2])
		end
		local out = params.model:forward(cur_input)
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
	for i = 1, #params.index2label do
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
print('data='..params.rep..';dev_score='..dev_fscore..';test_score='..test_fscore..';lr='..params.lr..';bsize='..params.batch_size..';dropout='..params.dropout_p..';epochs='..params.num_epochs)
print(res)