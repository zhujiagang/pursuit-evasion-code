--[[
Copyright (c) 2014 Google Inc.
See LICENSE file for full terms of limited license.
]]

gd = require "gd"
image = require "image"
if not dqn then
    require "initenv"
end
-- Gameindex=4

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
gif_file="../gifs/Evasion_Test.gif"
gpu=0
random_starts=30
pool_frms="type=\"max\",size=2"
num_threads=4
-- args="-name $agent_name -env_params $env_params -agent $agent -agent_params $agent_params -actrep $actrep -gpu $gpu -random_starts $random_starts -pool_frms $pool_frms -seed $seed -threads $num_threads -network $NETWORK -gif_file $gif_file"
-- print(args)

arg[1] = "-name"
arg[2] = agent_name
arg[3] = "-agent"
arg[4] = agent
arg[5] = "-agent_params"
arg[6] = agent_params
arg[7] = "-actrep"
arg[8] = actrep
arg[9] = "-gpu"
arg[10] = gpu
arg[11] = "-random_starts"
arg[12] = random_starts
arg[13] = "-pool_frms"
arg[14] = pool_frms
arg[15] = "-seed"
arg[16] = seed
arg[17] = "-threads"
arg[18] = num_threads

arg[19] = "-network"
arg[20] = NETWORK

arg[21] = "-gif_file"
arg[22] = gif_file

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

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')
cmd:option('-gif_file', '', 'GIF path to write session screens')
cmd:option('-csv_file', '', 'CSV path to write session data')

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

-- file names from command line
local gif_filename = opt.gif_file
-- start a new game
game_actions = {0,1,2,3,4}

local maxqq
local lasthid={}
qmaxtest={}
lasthidden={}
qnowtest={}
filename1 = "mylast"
filename2 = "qmaxtest"
filename3 = "qnowtest"

for Gameindex=3,3 do
Gameindex=4
parainit()

screen, reward, terminal,testscreen = NewGame(Gameindex,true)
image.save('G' .. Gameindex .. '.jpg',testscreen)
-- compress screen to JPEG with 100% quality
local jpg = image.compressJPG(screen:squeeze(), 100)
-- create gd image from JPEG string
local im = gd.createFromJpegStr(jpg:storage():string())
-- convert truecolor to palette
im:trueColorToPalette(false, 256)

-- write GIF header, use global palette and infinite looping
im:gifAnimBegin(gif_filename, true, 0)
-- write first frame
im:gifAnimAdd(gif_filename, false, 0, 0, 7, gd.DISPOSAL_NONE)

-- remember the image and show it first
local previm = im
local win = image.display({image=screen})

print("Started playing...",'Game:',Gameindex)
-- play one episode (game)

local i=0;
CaptureCnt=0
SurroundCnt=0
Maxiter=100000
while i< Maxiter do
    -- if action was chosen randomly, Q-value is 0
    i=i+1
    -- choose the best action
    local action_index ,lasthid,qtemp = agent:perceive(reward, screen, terminal, true, 0.05)    
    --local action_index =torch.random(1,5)
    -- filename1 = "lasthiddenlayer"
    -- lasthidden[i]=lasthid
    -- torch.save(filename1 .. i .. ".t7",lasthidden,'ascii')

    -- lasthidden[i]=lasthid
    -- torch.save(filename1 .. ".t7",lasthidden,'ascii')

    -- qmaxtest[i]=agent.bestq    
    -- torch.save(filename2 .. ".t7",qmaxtest,'ascii')

    -- qnowtest[i]=qtemp    
    -- torch.save(filename3 .. ".t7",qnowtest,'ascii')

    -- image.save(tostring(i) .. '.jpg',testscreen)

    -- play game in test mode (episodes don't end when losing a life)
    -- screen, reward, terminal = game_env:step(game_actions[action_index], false)
    screen, reward, terminal,Surround = EvasionGame(game_actions[action_index],Gameindex)
    if terminal then
        screen, reward, terminal = NextRandGame(Gameindex,30,true)
        CaptureCnt = CaptureCnt+1
    end
    if Surround then
        SurroundCnt = SurroundCnt+1
    end
    -- display screen
    image.display({image=screen, win=win})
    -- create gd image from tensor
    jpg = image.compressJPG(screen:squeeze(), 100)
    im = gd.createFromJpegStr(jpg:storage():string())
    
    -- use palette from previous (first) image
    im:trueColorToPalette(false, 256)
    im:paletteCopy(previm)

    -- write new GIF frame, no local palette, starting from left-top, 7ms delay
    im:gifAnimAdd(gif_filename, false, 0, 0, 7, gd.DISPOSAL_NONE)
    -- remember previous screen for optimal compression
    previm = im
end
-- CaptureSurround=torch.Tensor{CaptureCnt,SurroundCnt}
-- local filename4 = "CaptureSurround"
-- torch.save(filename4 .. ".t7",CaptureSurround,'ascii')

print('Game:',Gameindex,'SurroundCnt         :',SurroundCnt,'CaptureCnt:',CaptureCnt)

-- end GIF animation and close CSV file
gd.gifAnimEnd(gif_filename)

end

print("Finished playing, close window to exit!")