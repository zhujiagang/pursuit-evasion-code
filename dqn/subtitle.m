function ht = subtitle(kn,text,txtsize)
 h1 = get(gcf,'children');
 axis1 = get(h1(end),'Position');
 axis2 = get(h1(end-kn+1),'Position');
 axest = [axis1(1),axis1(2)+axis1(4),axis2(1)+axis1(3)-axis1(1),0.01];
 ht = axes('Position',axest);
 axis(ht,'off')
 title(ht,text,'FontSize',txtsize)