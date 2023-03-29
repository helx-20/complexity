function new_u=lif_model_multiple(C,u,U_rest,I)
    global dt;
    du=-(u-U_rest)./C+I./C;
    new_u=u+du*dt;
end