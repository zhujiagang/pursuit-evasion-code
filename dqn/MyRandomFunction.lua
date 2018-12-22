require 'gnuplot'
require 'torch'
require 'image'
require 'nn'

function myrandom(start,endt)

	local l11=tostring(sys.clock()):reverse():sub(1,6)

	l12=start+tonumber(torch.floor(l11))%(endt-start+1)

	return l12
end
