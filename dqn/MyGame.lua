require 'gnuplot'
require 'torch'
require 'image'
require 'nn'
require 'MyRandomFunction'


metric=84

XBound=metric/2
escape_speed=1
puesue_speed=escape_speed*2/3

initlen=4
gridiv=2
gridunit=1

grayescape=100
graypursue=200

escapeinit=4*initlen

terminalscope=1
separatescope=4

bound=false 		

function addlimit(r,a)
	r=r+a;
	if r>XBound  then
    	r=XBound;
    elseif r<-XBound+1 then
    	r=-XBound+1;
    end
    return r;
end


function escape(x,y,actionindex)
    local a,b
    if actionindex==0 then
    	a=0
    	b=0
    	escapeangle=4*math.pi
    else
	    a=escape_speed*math.cos(actionindex*math.pi/gridiv)
	    b=escape_speed*math.sin(actionindex*math.pi/gridiv)
	    escapeangle=actionindex*math.pi/gridiv
	end

    x=addlimit(x,a)
    y=addlimit(y,b)

	return x,y,escapeangle
end


function pursue(x1,y1,x,y,centerx,centery)

	local base =math.sqrt(math.pow(x-centerx,2)+math.pow(y-centery,2))
	local a,b

	if base == 0 then
		a=0;
		b=0;
	else
	    a=puesue_speed*(x-centerx)/base
	    b=puesue_speed*(y-centery)/base
	end

    x1=addlimit(x1,a)
    y1=addlimit(y1,b)

	return x1,y1
end

function leftrightlimit1(maxx,maxxid,minx,minxid,a,xy,xyx,l,r)
	local i
    if maxx+a>XBound  then
		for i=l,r do
			if xyx[i]~=maxxid then
				xy[xyx[i]][1]=xy[xyx[i]][1]+XBound-maxx
			end
		end
    	maxx=XBound; 

    elseif minx+a<-XBound+1  then
		for i=l,r do
			if xyx[i]~=minxid then
				xy[xyx[i]][1]=xy[xyx[i]][1]-XBound+1-minx
			end
		end
    	minx=-XBound+1;
    else
    	for i=l,r do
			xy[xyx[i]][1]=xy[xyx[i]][1]+a
		end
    end
    return xy
end


function leftrightlimit2(maxx,maxxid,minx,minxid,a,xy,xyx,l,r)
	local i
    if maxx+a>XBound  then
		for i=l,r do
			if xyx[i]~=maxxid then
				xy[xyx[i]][2]=xy[xyx[i]][2]+XBound-maxx
			end
		end
    	maxx=XBound;    

    elseif minx+a<-XBound+1  then
		for i=l,r do
			if xyx[i]~=minxid then
				xy[xyx[i]][2]=xy[xyx[i]][2]-XBound+1-minx
			end
		end
    	minx=-XBound+1;

    else
    	for i=l,r do
			xy[xyx[i]][2]=xy[xyx[i]][2]+a
		end
    end
    return xy
end

function leftrightbound(maxx,minx,a)
    if maxx+a>XBound  then
    	bound=true  	
    elseif minx+a<-XBound+1  then
    	bound=true  	
    end
end

function pursuebound(xy,x,y,centerx,centery,xyx,l,r)
	local i

	local base = math.sqrt(math.pow(x-centerx,2)+math.pow(y-centery,2))
	local a,b
	if base == 0 then
		a=0;
		b=0;
	else
	    a=puesue_speed*(x-centerx)/base
	    b=puesue_speed*(y-centery)/base
	end

	maxidx=xyx[l]
	maxidy=xyx[l]
	minidx=xyx[l]
	minidy=xyx[l]
	maxx,maxy,minx,miny=xy[xyx[l]][1],xy[xyx[l]][2],xy[xyx[l]][1],xy[xyx[l]][2]

	for i=l+1,r do
		if maxx<xy[xyx[i]][1] then
			maxidx=xyx[i]
			maxx=xy[xyx[i]][1]
		elseif minx>xy[xyx[i]][1] then
			minidx=xyx[i]
			minx=xy[xyx[i]][1]
		end
		if maxy<xy[xyx[i]][2] then
			maxidy=xyx[i]
			maxy=xy[xyx[i]][2]
		elseif miny>xy[xyx[i]][2] then
			minidy=xyx[i]
			miny=xy[xyx[i]][2]
		end
	end
	bound=false
	leftrightbound(maxx,minx,a)
	leftrightbound(maxy,miny,b)
	return bound
end



function pursuetogether(xy,x,y,centerx,centery,xyx,l,r)
	local i
	local base =math.sqrt(math.pow(x-centerx,2)+math.pow(y-centery,2))
	local a,b
	if base == 0 then
		a=0;
		b=0;
	else
	    a=puesue_speed*(x-centerx)/base
	    b=puesue_speed*(y-centery)/base
	end

	maxidx=xyx[l]
	maxidy=xyx[l]
	minidx=xyx[l]
	minidy=xyx[l]
	maxx,maxy,minx,miny=xy[xyx[l]][1],xy[xyx[l]][2],xy[xyx[l]][1],xy[xyx[l]][2]

	for i=l+1,r do
		if maxx<xy[xyx[i]][1] then
			maxidx=xyx[i]
			maxx=xy[xyx[i]][1]
		elseif minx>xy[xyx[i]][1] then
			minidx=xyx[i]
			minx=xy[xyx[i]][1]
		end
		if maxy<xy[xyx[i]][2] then
			maxidy=xyx[i]
			maxy=xy[xyx[i]][2]
		elseif miny>xy[xyx[i]][2] then
			minidy=xyx[i]
			miny=xy[xyx[i]][2]
		end
	end

	xy=leftrightlimit1(maxx,maxxid,minx,minxid,a,xy,xyx,l,r)
	xy=leftrightlimit2(maxy,maxyid,miny,minyid,b,xy,xyx,l,r)

	return xy
end

function body1(x,y,xxx,id)
	yb=XBound+1
	xxx[x+yb][y+yb] = id
	xxx[x+yb+1][y+yb] = id
	xxx[x+yb-1][y+yb] = id
	xxx[x+yb][y+yb+1] = id
	xxx[x+yb][y+yb-1] = id

	return xxx
end

function body2(x,y,xxx,id)
	yb=XBound+1
	xxx[x+yb][y+yb] = id
	xxx[x+yb+1][y+yb] = id
	xxx[x+yb-1][y+yb] = id
	xxx[x+yb][y+yb+1] = id
	xxx[x+yb][y+yb-1] = id

	xxx[x+yb+1][y+yb-1] = id
	xxx[x+yb+1][y+yb+1] = id
	xxx[x+yb-1][y+yb+1] = id
	xxx[x+yb-1][y+yb-1] = id
	return xxx
end

function imgsave()
	xxx = torch.zeros(metric+4, metric+4)
	xxx = body1(x,y,xxx,grayescape)

	for i=1,iiii do
		xxx = body2(xy[i][1],xy[i][2],xxx,graypursue)
	end

	return xxx
end

function body(x,y,xx,id)
	xx[x+XBound][y+XBound] = id
	-- xx[x+XBound+1+1][y+XBound+1] = id
	-- xx[x+XBound-1+1][y+XBound+1] = id
	-- xx[x+XBound+1][y+XBound+1+1] = id
	-- xx[x+XBound+1][y+XBound-1+1] = id

	-- xx[x+XBound+1][y+XBound-1] = id
	-- xx[x+XBound+1][y+XBound+1] = id
	-- xx[x+XBound-1][y+XBound+1] = id
	-- xx[x+XBound+1][y+XBound+1] = id
	return xx
end




function imgdisplay()
	local i
	
	xx = torch.zeros(metric, metric)
	xx=body(x,y,xx,grayescape)

	for i=1,iiii do
		xx=body(xy[i][1],xy[i][2],xx,graypursue)
	end

	return xx
end

function parainit()
	local i,j
	Sigma=1;
	GradNow=0;
	GradLast=0;

	GradWalltemp=torch.zeros(XBound,1);
	GradWall=torch.zeros(2*XBound,2*XBound);

	for i=-XBound+1,0 do		
		GradWalltemp[i+XBound][1]=math.exp(-(i+XBound)*(i+XBound)/(2*Sigma*Sigma))/(Sigma*math.sqrt(2*math.pi));
	end

	for i=-XBound+1,XBound do
		for j=-XBound+1,XBound do
			if j<=0 and j>=-XBound+1 then
				GradWall[i+XBound][j+XBound]=GradWall[i+XBound][j+XBound]+GradWalltemp[j+XBound][1]
			end
			if j>0 and j<=XBound then
				GradWall[i+XBound][j+XBound]=GradWall[i+XBound][j+XBound]+GradWalltemp[-j+XBound+1][1]
			end
		end
	end

	for j=-XBound+1,XBound do
		for i=-XBound+1,XBound do
			if i<=0 and i>=-XBound+1 then
				GradWall[i+XBound][j+XBound]=GradWall[i+XBound][j+XBound]+GradWalltemp[i+XBound][1]
			end
			if i>0 and i<=XBound then
				GradWall[i+XBound][j+XBound]=GradWall[i+XBound][j+XBound]+GradWalltemp[-i+XBound+1][1]
			end
		end
	end
end



function NewGameInit(index,testing)
	local i,j
	if index==1 then
		-- local iii=torch.random(0,3);
		local iii
		if not testing then
			iii=myrandom(0,2);
		else
			iii=3
		end
		x=escapeinit*math.cos(iii*math.pi/gridiv)
		y=escapeinit*math.sin(iii*math.pi/gridiv)
		iiii=4
		xy = torch.zeros(iiii, 2)
		xy[1][1] = initlen
		xy[1][2] = initlen
		xy[2][1] = initlen
		xy[2][2] = -initlen
		xy[3][1] = -initlen
		xy[3][2] = initlen
		xy[4][1] = -initlen
		xy[4][2] = -initlen
	elseif index==2 then
		-- local iii=torch.random(0,3);
		-- local iii=myrandom(0,3);
		local iii
		if not testing then
			iii=myrandom(1,3);
		else
			iii=0
		end
		x=escapeinit*math.cos(iii*math.pi/gridiv)
		y=escapeinit*math.sin(iii*math.pi/gridiv)

		iiii=myrandom(4,8);

		xy = torch.zeros(iiii, 2)

		xy[1][1]=torch.random(-initlen,initlen);
		xy[1][2]=torch.random(-initlen,initlen);
		for i=2,iiii do
			xy[i][1]=torch.random(-initlen,initlen);
			xy[i][2]=torch.random(-initlen,initlen);
			for j=1,i-1 do
				if xy[i][1]==xy[j][1] and xy[i][2]==xy[j][2] then
				-- xy[i][1]=torch.random(-initlen,initlen);
				xy[i][1]=torch.random(-initlen,initlen);
				xy[i][2]=torch.random(-initlen,initlen);
				j=1;
				end
			end
		end
	elseif index==3 or index==4 or index==5 then
		if index==4 then 
			iiii=4
		else 
			iiii=myrandom(4,8)
		end
		xy = torch.zeros(iiii, 2)
		unready=true
		while unready do
			xy[1][1]=torch.random(-XBound+1,XBound);
			xy[1][2]=torch.random(-XBound+1,XBound);
			for i=2,iiii do
				xy[i][1]=torch.random(-XBound+1,XBound);
				xy[i][2]=torch.random(-XBound+1,XBound);
				for j=1,i-1 do
					if xy[i][1]==xy[j][1] and xy[i][2]==xy[j][2] then
						xy[i][1]=torch.random(-XBound+1,XBound);
						xy[i][2]=torch.random(-XBound+1,XBound);
						j=1;
					end
					if xy[i][1]*xy[i][1] + xy[i][2]*xy[i][2]<XBound*XBound/4 then
						xy[i][1]=torch.random(-XBound+1,XBound);
						xy[i][2]=torch.random(-XBound+1,XBound);
						j=1;
					end
				end
			end

			local xtemp=0
			local ytemp=0
			local t
			for t=1,iiii do
				xtemp=xtemp+xy[t][1];
				ytemp=ytemp+xy[t][2];
			end
			x=torch.floor(xtemp/iiii)
			y=torch.floor(ytemp/iiii)
			unready=false
			for j=1,iiii do
				if x==xy[j][1] and y==xy[j][2] then
					unready=true
					break
				end
			end
			if x>0 and y>0 and not testing then
				unready=true
			end
			if (x<0 or y<0) and testing then
				unready=true
			end
		end
	elseif index==6 then
		iiii=myrandom(4,8)
		if not testing then
			unready=true
			while unready do
				x=torch.random(-XBound/2+1,XBound/2);
				y=torch.random(-XBound/2+1,XBound/2);
				unready=false
				if x<0 and y<0 then
					unready=true
				end
			end
		else
			while true do
				x=torch.random(-XBound/2+1,XBound/2);
				y=torch.random(-XBound/2+1,XBound/2);
				if x<0 and y<0 then
					break
				end
			end
		end

		unready=true
		while unready do
			xy = torch.zeros(iiii, 2)
			xy[1][1]=torch.random(-XBound+1,XBound)
			xy[1][2]=torch.random(-XBound+1,XBound)
			for i=2,iiii do
				xy[i][1]=torch.random(-XBound+1,XBound);
				xy[i][2]=torch.random(-XBound+1,XBound);
				for j=1,i-1 do
					if xy[i][1]==xy[j][1] and xy[i][2]==xy[j][2] then
						xy[i][1]=torch.random(-XBound+1,XBound);
						xy[i][2]=torch.random(-XBound+1,XBound);
						j=1;
					end
				end
			end
			unready=false
			for j=1,iiii do
				if x==xy[j][1] and y==xy[j][2] then
					unready=true
				end
			end
		end
	end
	
	Terminal=false;
	Reward=0;
	ImageCnt=0;

	xx=imgdisplay()

	-- xxx=imgsave()
	-- return xx,Reward,Terminal--,xxx
end


function GameSetting(actionindex,index)
	local i
	x,y,eangle=escape(x,y,actionindex)
	xyx = torch.zeros(iiii)
	xyxid = torch.zeros(iiii)
	for i=1,iiii do
		xyx[i]=(xy[i][1]-x)*(xy[i][1]-x)+(xy[i][2]-y)*(xy[i][2]-y)
		xyxid[i]=i
	end
	i=1
	for i = 1, iiii do
        for j = i+1,iiii do
            if (xyx[i] > xyx[j]) then
                local temp = xyx[i];
                xyx[i] = xyx[j];
                xyx[j] = temp;

				temp = xyxid[i];
                xyxid[i] = xyxid[j];
                xyxid[j] = temp;
            end
        end
    end
	l=torch.min(xy,1)
	r=torch.max(xy,1)
    left=l[1][1]
    down=l[1][2]
    right=r[1][1]
    up=r[1][2]
    if index==1 or index==2 then
    	bound1=false
    	if index==2 then 
    		bound1=pursuebound(xy,x,y,xy[xyxid[1]][1],xy[xyxid[1]][2],xyxid,1,iiii)
    	end
    	if (x<right+1 and x>left-1 and y<up+1 and y>down-1) or bound1 then 
    		i=1
			for i=1,iiii do
				xy[i][1],xy[i][2]=pursue(xy[i][1],xy[i][2],x,y,xy[i][1],xy[i][2])
			end
    	else
    		xy=pursuetogether(xy,x,y,xy[xyxid[1]][1],xy[xyxid[1]][2],xyxid,1,iiii)
    	end
    elseif index==3 then
    	i=1
		for i=1,iiii do
			xy[i][1],xy[i][2]=pursue(xy[i][1],xy[i][2],x,y,xy[i][1],xy[i][2])
		end
	elseif index==4 or index==5 or index==6 then

		bound1=pursuebound(xy,x,y,xy[xyxid[1]][1],xy[xyxid[1]][2],xyxid,1,torch.floor(iiii/2))
		tt=torch.floor(iiii/2)+1
    	bound2=pursuebound(xy,x,y,xy[xyxid[tt]][1],xy[xyxid[tt]][2],xyxid,tt,iiii)

		if bound1 or bound2 then
			i=1
	  		for i=1,iiii do
				xy[i][1],xy[i][2]=pursue(xy[i][1],xy[i][2],x,y,xy[i][1],xy[i][2])
			end
		else
			if (x<right and x>left and y<up and y>down) then 
				i=1
				for i=1,iiii do
					xy[i][1],xy[i][2]=pursue(xy[i][1],xy[i][2],x,y,xy[i][1],xy[i][2])
				end
				first=1
		    else 
				local tempx=0
			    local tempy=0
			    i=1
				for i=1,torch.floor(iiii/2) do
					tempx=tempx+xy[xyxid[i]][1]
					tempy=tempy+xy[xyxid[i]][2]
				end

				local centerx=tempx/torch.floor(iiii/2);
				local centery=tempy/torch.floor(iiii/2);

			    local tempx=0
			    local tempy=0
			    i=torch.floor(iiii/2)+1
				for i=torch.floor(iiii/2)+1,iiii do
					tempx=tempx+xy[xyxid[i]][1]
					tempy=tempy+xy[xyxid[i]][2]
				end

				local centerx1=tempx/(iiii-torch.floor(iiii/2));
				local centery1=tempy/(iiii-torch.floor(iiii/2));

			    xy=pursuetogether(xy,x,y,xy[xyxid[1]][1],xy[xyxid[1]][2],xyxid,1,torch.floor(iiii/2))

		    	if math.pow(centerx1-centerx,2)+math.pow(centery1-centery,2)<separatescope then
					first=1
		    	end

				if first==1 then
					tt=torch.floor(iiii/2)+1
			    	xy=pursuetogether(xy,0,0,centerx1,centery1,xyxid,tt,iiii)
				    if math.pow(centerx1,2)+math.pow(centery1,2) <= 1 then 
				    	first=0
				    end
				else
					tt=torch.floor(iiii/2)+1
			    	xy=pursuetogether(xy,x,y,xy[xyxid[tt]][1],xy[xyxid[tt]][2],xyxid,tt,iiii)
			    end
			end
		end
	end



    mint=math.pow(x-xy[1][1],2)+math.pow(y-xy[1][2],2)
	for i=2,iiii do
		if math.pow(x-xy[i][1],2)+math.pow(y-xy[i][2],2)<mint then
			mint=math.pow(x-xy[i][1],2)+math.pow(y-xy[i][2],2)
		end
	end

	GradLast=GradNow;
	GradNow=0;
	local kk
	for kk=1,iiii do
		GradNow=GradNow+math.exp(-(math.pow(xy[kk][1]-x,2)+math.pow(xy[kk][2]-y,2))/(2*Sigma*Sigma))/(2*math.pi*Sigma*Sigma);
	end

	GradNow=GradNow+(math.exp(-(math.pow(x+XBound-1,2))/(2*Sigma*Sigma))
	+math.exp(-(math.pow(y+XBound-1,2))/(2*Sigma*Sigma))
	+math.exp(-(math.pow(XBound-x,2))/(2*Sigma*Sigma))
	+math.exp(-(math.pow(XBound-y,2))/(2*Sigma*Sigma)))/(2*math.pi*Sigma*Sigma)
	Surround=false
	if (x<right and x>left and y<up and y>down) then
		Surround=true
		local differforcex=0
		local differforcey=0
		for i=1,iiii do 
			local r2=math.pow(x-xy[i][1],2)+math.pow(y-xy[i][2],2)
			if r2 == 0 then
				Terminal=true
				break
			else
				differforcex=differforcex+(x-xy[i][1])/(r2*r2)
				differforcey=differforcey+(y-xy[i][2])/(r2*r2)
			end
		end
		
		differforcex=differforcex+1*((x-XBound)/(math.pow(x-XBound,4)+0.0001)+(x+XBound)/(math.pow(x+XBound,4)+0.0001))
		differforcey=differforcey+1*((y-XBound)/(math.pow(y-XBound,4)+0.0001)+(y+XBound)/(math.pow(y+XBound,4)+0.0001))

		local dangle=torch.atan(differforcey/differforcex)
		if differforcey<0 and differforcex>0 then
			dangle=dangle+2*math.pi
		elseif differforcey<0 and differforcex<0 then
			dangle=dangle+math.pi
		elseif differforcey>0 and differforcex<0 then
			dangle=dangle+math.pi
		end
        if eangle>=dangle-math.pi/4 and eangle<=dangle+math.pi/4 then
        	Reward=1;
        else
        	Reward=-1;
        end
	elseif (x==XBound or x==-XBound+1 or y==XBound or y==-XBound+1) then 
		Reward=-1;
	elseif (mint<initlen*initlen) then
		Reward=-1;
	elseif mint>4*initlen*initlen and GradNow<GradLast then
		Reward=1;
	else
		Reward=0;
	end

	if (mint<terminalscope) then
		Terminal=true;
	end
end

function NewGame(index,testing)
	NewGameInit(index,testing)
	--xxx=imgsave()
	-- return xx,Reward,Terminal--,xxx
	return xx,Reward,Terminal--,xxx
end

function NextRandGame(index,inde,testing)

	while true do
		NewGameInit(index,testing)
		local k=myrandom(1,inde)
		for kk=1,k do
			GameSetting(0,index)
			ImageCnt=ImageCnt+1
			xx=imgdisplay()
			if Terminal then
				-- print(string.format('WARNING: Terminal signal received after %d 0-steps',kk))
				break
			end
		end
		if not Terminal then
			break
		end
	end

	return xx,Reward,Terminal,Surround 
end

function EvasionGame(actionindex,index)
	GameSetting(actionindex,index)
	ImageCnt=ImageCnt+1
	xx=imgdisplay()
	-- xxx=imgsave()
	return xx,Reward,Terminal,Surround 
end