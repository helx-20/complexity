function new_u=lif_simplified(C,u,U_rest,I)
    global dt;
    decay=0.9;
    du=-(u-U_rest)*(1-decay)/C+I/C;
    new_u=u+du*dt;
end