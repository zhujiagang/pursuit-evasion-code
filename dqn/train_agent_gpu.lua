require "initenv"

Gameindex=4

NETWORK="DQN3_0_1_Evasion_FULL_Y.t7"
agent="NeuralQLearner"
n_replay=1
netfile="\"convnet_evasion\""
update_freq=4
actrep=4
discount=0.99
seed=1
learn_start=50000
pool_frms_type="\"max\""
pool_frms_size=2
initial_priority="false"
replay_memory=1000000
eps_end=0.1
eps_endt=replay_memory
lr=0.00025
agent_type="DQN3_0_1"
preproc_net="\"net_downsample_2x_full_y\""
agent_name="DQN3_0_1_Evasion_FULL_Y"
state_dim=7056
ncols=1

agent_params="lr=" .. lr .. ",ep=1,ep_end=" .. eps_end .. ",ep_endt=" .. eps_endt .. ",discount=" .. discount .. ",hist_len=4,learn_start=" .. learn_start .. ",replay_memory=" .. replay_memory .. ",update_freq=" .. update_freq .. ",n_replay=" .. n_replay .. ",network=" .. netfile .. ",preproc=" .. preproc_net .. ",state_dim=" .. state_dim .. ",minibatch_size=32,rescale_r=1,ncols=" .. ncols .. ",bufferSize=512,valid_size=500,target_q=10000,clip_delta=1,min_reward=-1,max_reward=1"
steps=6000000

prog_freq=50000
save_freq=50000

eval_freq=50000
eval_steps=50000

gpu=0
random_starts=30
pool_frms="type=\"max\",size=2"
num_threads=4

-- args="-name " .. agent_name .. " -agent " .. agent .. " -agent_params " .. agent_params .. " -steps " .. steps .. " -eval_freq " .. eval_freq .. " -eval_steps " .. eval_steps .. " -prog_freq " .. prog_freq .. " -save_freq " .. save_freq .. " -actrep " .. actrep .. " -gpu " .. gpu .. " -random_starts " .. random_starts .. " -pool_frms " .. pool_frms .. " -seed " .. seed .. " -threads " .. num_threads
-- print(args)

arg[1] = "-name"
arg[2] = agent_name
arg[3] = "-agent"
arg[4] = agent
arg[5] = "-agent_params"
arg[6] = agent_params
arg[7] = "-steps"
arg[8] = steps
arg[9] = "-eval_freq"
arg[10] = eval_freq
arg[11] = "-eval_steps"
arg[12] = eval_steps
arg[13] = "-prog_freq"
arg[14] = prog_freq

arg[15] = "-save_freq"
arg[16] = save_freq
arg[17] = "-actrep"
arg[18] = actrep
arg[19] = "-gpu"
arg[20] = gpu
arg[21] = "-random_starts"
arg[22] = random_starts
arg[23] = "-pool_frms"
arg[24] = pool_frms
arg[25] = "-seed"
arg[26] = seed
arg[27] = "-threads"
arg[28] = num_threads
arg[29] = "-network"
arg[30] = NETWORK

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')

cmd:text()

local opt = cmd:parse(arg)

--- General setup.
local agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local step_history = {}
local maxqrewardstep={}
local step = 0
local qmax_ave_history={}
time_history[1] = 0

local total_reward
local nrewards
local nepisodes
local episode_reward
local total_qmax

game_actions = {0,1,2,3,4}

parainit()

print('Game:',Gameindex)
local screen, reward, terminal = NewGame(Gameindex)
local action_index=0
print("Iteration ..", step)
local win = nil

local SnapshotCnt=0

if Snapshot and arg[29] and arg[30] then
    Snapshot=false
    local msg, err = pcall(require, NETWORK)
    if not msg then
        -- try to load saved agent
        -- print(self.network)
        local err_msg, exp = pcall(torch.load, NETWORK)
        if not err_msg then
            error("Could not find network file ")
        end 
        if exp.model and exp.reward_history and exp.step_history and exp.maxqrewardstep and exp.step and exp.qmax_ave_history 
            and exp.time_history then
                Snapshot=true
                reward_history = exp.reward_history
                step_history = exp.step_history
                maxqrewardstep = exp.maxqrewardstep
                step = exp.step
                qmax_ave_history = exp.qmax_ave_history
                time_history = exp.time_history
        else
            reward_history = {}
            step_history = {}
            maxqrewardstep={}
            step = 0
            qmax_ave_history={}
        end
    end
end

print('Snapshot',Snapshot,'step',step)

while step < opt.steps do
    if Snapshot then
        SnapshotCnt = SnapshotCnt+1
        -- print(SnapshotCnt)        
        action_index = agent:perceive(reward, screen, terminal)
        -- game over? get next game!
        if not terminal then
            -- screen, reward, terminal = game_env:step(game_actions[action_index], true)
            screen, reward, terminal = EvasionGame(game_actions[action_index],Gameindex)
        else
            -- print(opt.random_starts)
            if opt.random_starts > 0 then
                -- screen, reward, terminal = NewGame(Gameindex)
                screen, reward, terminal = NextRandGame(Gameindex,opt.random_starts)
            else
                screen, reward, terminal = NewGame(Gameindex)
            end
        end
        -- display screen
        win = image.display({image=screen, win=win})
        if SnapshotCnt > learn_start then
            Snapshot = false
            step = step + 1
        end
    else
        step = step + 1
        action_index = agent:perceive(reward, screen, terminal)
        -- print(step)
        -- game over? get next game!
        if not terminal then
            -- screen, reward, terminal = game_env:step(game_actions[action_index], true)
            screen, reward, terminal = EvasionGame(game_actions[action_index],Gameindex)
        else
            if opt.random_starts > 0 then
                -- screen, reward, terminal = NewGame(Gameindex)
                screen, reward, terminal = NextRandGame(Gameindex,opt.random_starts)
            else
                screen, reward, terminal = NewGame(Gameindex)
            end
        end
        -- print(screen:size())
        -- display screen
        win = image.display({image=screen, win=win})

        if step % opt.prog_freq == 0 then
            assert(step==agent.numSteps, 'trainer step: ' .. step ..
                    ' & agent.numSteps: ' .. agent.numSteps)
            print("Steps: ", step)
            print('Game:',Gameindex)
            agent:report()
            collectgarbage()
        end

        if step%1000 == 0 then collectgarbage() end

        if step % opt.eval_freq == 0 and step > learn_start then

            screen, reward, terminal = NewGame(Gameindex,true)
            total_reward = 0
            total_qmax=0
            nrewards = 0
            nepisodes = 0
            episode_reward = 0
            local maxreward=0
            local maxstep=0
            local maxq=0
            local eval_time = sys.clock()
            local cntstep=0
            for estep=1,opt.eval_steps do
                action_index = agent:perceive(reward, screen, terminal, true, 0.05)
                -- Play game in test mode (episodes don't end when losing a life)
                -- screen, reward, terminal = game_env:step(game_actions[action_index])         
                -- display screen
                win = image.display({image=screen, win=win})
                if not terminal then
                    -- screen, reward, terminal = game_env:step(game_actions[action_index], true)
                    screen, reward, terminal = EvasionGame(game_actions[action_index],Gameindex)               
                end

                if estep%1000 == 0 then collectgarbage() end

                -- record every reward
                episode_reward = episode_reward + reward
                if reward ~= 0 then
                   nrewards = nrewards + 1
                end         

                cntstep=cntstep+1

                if terminal then
                    total_reward = total_reward + episode_reward
                    
                    nepisodes = nepisodes+1
                    if cntstep>maxstep then
                        maxstep=cntstep
                    end

                    if episode_reward>maxreward then
                        maxreward=episode_reward
                    end

                    cntstep=0
                    episode_reward = 0

                    -- screen, reward, terminal = NewGame(Gameindex)
                    screen, reward, terminal = NextRandGame(Gameindex,opt.random_starts,true)
                end
                if maxq<agent.bestq then
                    maxq=agent.bestq
                end

                total_qmax = total_qmax + agent.bestq
            end

            if nepisodes==0 then
                total_reward = episode_reward

                maxstep=cntstep
                maxreward=episode_reward

                episode_reward = 0
                nepisodes = 1
            end

            eval_time = sys.clock() - eval_time
            start_time = start_time + eval_time
            -- agent:compute_validation_statistics()
            local ind = #reward_history+1

            total_reward = total_reward/math.max(1, nepisodes)

            total_qmax = total_qmax/opt.eval_steps

            if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
                agent.best_network = agent.network:clone()
            end

            -- if agent.v_avg then
            --     v_history[ind] = agent.v_avg
            --     td_history[ind] = agent.tderr_avg
            --     qmax_history[ind] = agent.q_max
            -- end

            -- print("V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])

            reward_history[ind] = total_reward
            local filename1 = "reward_history"
            torch.save(filename1 .. ".t7",reward_history,'ascii')

            step_history[ind] = opt.eval_steps/nepisodes
            local filename2 = "step_history"
            torch.save(filename2 .. ".t7",step_history,'ascii')

            qmax_ave_history[ind] = total_qmax
            local filename3 = "qmax"
            torch.save(filename3 .. ".t7",qmax_ave_history,'ascii')

            maxqrewardstep[ind]=torch.Tensor{maxq,maxreward,maxstep}
            local filename4 = "maxqrewardstep"
            torch.save(filename4 .. ".t7",maxqrewardstep,'ascii')

            reward_counts[ind] = nrewards
            episode_counts[ind] = nepisodes

            time_history[ind+1] = sys.clock() - start_time

            local time_dif = time_history[ind+1] - time_history[ind]

            local training_rate = opt.actrep*opt.eval_freq/time_dif

            print(string.format(
                '\nSteps: %d (frames: %d), reward: %.2f, epsilon: %.2f, lr: %G, ' ..
                'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
                'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d',
                step, step*opt.actrep, total_reward, agent.ep, agent.lr, time_dif,
                training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
                nepisodes, nrewards))
        end

        if step % opt.save_freq == 0 or step == opt.steps then
            print('Game:',Gameindex)
          --   if step % 100 == 0 or step == opt.steps then
            local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
                agent.valid_s2, agent.valid_term
            agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
                agent.valid_term = nil, nil, nil, nil, nil, nil, nil
            local w, dw, g, g2, delta, delta2, deltas, tmp = agent.w, agent.dw,
                agent.g, agent.g2, agent.delta, agent.delta2, agent.deltas, agent.tmp
            agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
                agent.deltas, agent.tmp = nil, nil, nil, nil, nil, nil, nil, nil

            local filename = opt.name
            if opt.save_versions > 0 then
                filename = filename .. "_" .. math.floor(step / opt.save_versions)
            end
            filename = filename
            torch.save(filename .. ".t7", {agent = agent,
                                    model = agent.network,
                                    best_model = agent.best_network,
                                    reward_history = reward_history,
                                    step_history = step_history,
                                    qmax_ave_history=qmax_ave_history,
                                    maxqrewardstep=maxqrewardstep,
                                    reward_counts = reward_counts,
                                    episode_counts = episode_counts,
                                    time_history = time_history,
                                    v_history = v_history,
                                    td_history = td_history,
                                    qmax_history = qmax_history,
                                    arguments=opt,
                                    step=step
                                    })

            if opt.saveNetworkParams then
                local nets = {network=w:clone():float()}
                torch.save(filename..'.params.t7', nets, 'ascii')
            end
            agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
                agent.valid_term = s, a, r, s2, term
            agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
                agent.deltas, agent.tmp = w, dw, g, g2, delta, delta2, deltas, tmp
            print('Saved:', filename .. '.t7')
            io.flush()
            collectgarbage()
        end
    end
end