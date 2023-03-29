function main()%xor
    close all;
    global dt;
    global gk;
    global gna;
    global gl;
    global C;
    dt=0.001;%s
    t=40;%s
    
    lif1_1(t/dt)=0;
    lif1_2(t/dt)=0;
    lif1_3(t/dt)=0;
    lif1_4(t/dt)=0;
    
    lif3_1(t/dt)=0;
    lif3_2(t/dt)=0;
    lif3_3(t/dt)=0;
    lif3_4(t/dt)=0;
    
    lif2_1(t/dt)=0;
    lif2_2(t/dt)=0;
    lif2_3(t/dt)=0;
    lif2_4(t/dt)=0;
    
    output1(t/dt-1)=0;
    output2(t/dt-1)=0;
    
    U1_rest=50;
    U2_rest=-77;
    U3_rest=-54.387;
    U4_rest=0;
    
    gna = 120; gk = 36; gl = 0.3;
    gs=gna+gk+gl;
    
    hh1(t/dt,4)=0;%u,m,n,h
    hh2(t/dt,4)=0;
    
    % INITIALIZE STATE VARIABLES
    lif1_1(1)=0;
    lif1_2(1)=0;
    lif1_3(1)=0;
    lif1_4(1)=-70;
    f1=0;
    f2=0;
    f3=0;
    
    lif2_1(1)=0;
    lif2_2(1)=0;
    lif2_3(1)=0;
    lif2_4(1)=-70;
    f4=0;
    f5=0;
    f6=0;
    
    lif3_1(1)=0;
    lif3_2(1)=0;
    lif3_3(1)=0;
    lif3_4(1)=-70;
    f7=0;
    f8=0;
    f9=0;
    
    hh1(1,1)=-70;
    hh1(1,2)=0.0289;
    hh1(1,3)=0.2446;
    hh1(1,4)=0.7541;
    hh2(1,1)=-70;
    hh2(1,2)=0.0289;
    hh2(1,3)=0.2446;
    hh2(1,4)=0.7541; 
    hh3(1,1)=-70;
    hh3(1,2)=0.0289;
    hh3(1,3)=0.2446;
    hh3(1,4)=0.7541; 
    
    x1=@(t) 40;
    x2=@(t) 40;
    
    thresh=-40;
    W12=20;%weight
    C=1;
    
    for i=2:(t/dt)
        [hh1(i,1),hh1(i,2),hh1(i,3),hh1(i,4)]=hh_model(hh1(i-1,1),hh1(i-1,2),hh1(i-1,3),hh1(i-1,4),x1((i-1)*dt));
        if(hh1(i-1,1)>-40)
            spike_hh1=1;
        else
            spike_hh1=0;
        end
        gs=gna+gk+gl;
        lif1_1(i)=lif_model(C,lif1_1(i-1),U1_rest,x1((i-1)*dt)/gs);%C,u,U_rest,I
        lif1_2(i)=lif_model(C,lif1_2(i-1),U2_rest,x1((i-1)*dt)/gs);
        lif1_3(i)=lif_model(C,lif1_3(i-1),U3_rest,x1((i-1)*dt)/gs);
        f1=f1+(lif1_1(i)-lif1_1(i-1))*gna;
        f2=f2+(lif1_2(i)-lif1_2(i-1))*gk;
        f3=f3+(lif1_3(i)-lif1_3(i-1))*gl;
        lif1_4(i)=lif_model(C/gs,lif1_4(i-1),U4_rest,-(f1+f2+f3)+(gna*lif1_1(i-1)+gk*lif1_2(i-1)+gl*lif1_3(i-1))/gs);
        if(f1+f2+f3+lif1_4(i-1)>thresh)
            spike_lif1=1;
        else
            spike_lif1=0;
        end
        
        [hh3(i,1),hh3(i,2),hh3(i,3),hh3(i,4)]=hh_model(hh3(i-1,1),hh3(i-1,2),hh3(i-1,3),hh3(i-1,4),x2((i-1)*dt));
        if(hh3(i-1,1)>thresh)
            spike_hh3=1;
        else
            spike_hh3=0;
        end
        gs=gna+gk+gl;
        lif3_1(i)=lif_model(C,lif3_1(i-1),U1_rest,x2((i-1)*dt)/gs);%C,u,U_rest,I
        lif3_2(i)=lif_model(C,lif3_2(i-1),U2_rest,x2((i-1)*dt)/gs);
        lif3_3(i)=lif_model(C,lif3_3(i-1),U3_rest,x2((i-1)*dt)/gs);
        f7=f7+(lif3_1(i)-lif3_1(i-1))*gna;
        f8=f8+(lif3_2(i)-lif3_2(i-1))*gk;
        f9=f9+(lif3_3(i)-lif3_3(i-1))*gl;
        lif3_4(i)=lif_model(C/gs,lif3_4(i-1),U4_rest,-(f7+f8+f9)+(gna*lif3_1(i-1)+gk*lif3_2(i-1)+gl*lif3_3(i-1))/gs);
        if(f7+f8+f9+lif3_4(i-1)>thresh)
            spike_lif3=1;
        else
            spike_lif3=0;
        end
        
        [hh2(i,1),hh2(i,2),hh2(i,3),hh2(i,4)]=hh_model(hh2(i-1,1),hh2(i-1,2),hh2(i-1,3),hh2(i-1,4),spike_hh1*W12-spike_hh3*W12);
        if(hh2(i-1,1)>thresh)
            output1(i-1)=1;
        else
            output1(i-1)=0;
        end
        gs=gna+gk+gl;
        lif2_1(i)=lif_model(C,lif2_1(i-1),U1_rest,(W12*spike_lif1-W12*spike_lif3)/gs);
        lif2_2(i)=lif_model(C,lif2_2(i-1),U2_rest,(W12*spike_lif1-W12*spike_lif3)/gs);
        lif2_3(i)=lif_model(C,lif2_3(i-1),U3_rest,(W12*spike_lif1-W12*spike_lif3)/gs);
        f4=f4+(lif2_1(i)-lif2_1(i-1))*gna;
        f5=f5+(lif2_2(i)-lif2_2(i-1))*gk;
        f6=f6+(lif2_3(i)-lif2_3(i-1))*gl;
        lif2_4(i)=lif_model(C/gs,lif2_4(i-1),U4_rest,-(f4+f5+f6)+(gna*lif2_1(i-1)+gk*lif2_2(i-1)+gl*lif2_3(i-1))/gs);
        if(f4+f5+f6+lif2_4(i-1)>thresh)
            output2(i-1)=1;
        else
            output2(i-1)=0;
        end
    end
    save("data.mat",'output1','output2')
    t_current=dt*2000:dt*1000:t*1000;
    plot(t_current,output1);
    xlabel('time (ms)');
    ylabel('membrane(mV)');
    hold on;
    plot(t_current,output2);
    legend("hh","lif-hh")
    set(gca, 'xticklabel', get(gca, 'xtick'), 'yticklabel', get(gca, 'ytick'));
end