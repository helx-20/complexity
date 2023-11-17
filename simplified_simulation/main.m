function main()
    close all;
    global dt;
    global gk;
    global gna;
    global gl;
    global C;
    dt1=0.001;
    dt2=1;
    dt=dt1;%s
    t=80;%s
    
    lif1_1(t/dt2)=0;
    lif1_2(t/dt2)=0;
    lif1_3(t/dt2)=0;
    lif1_4(t/dt2)=0;
    
    lif2_1(t/dt2)=0;
    lif2_2(t/dt2)=0;
    lif2_3(t/dt2)=0;
    lif2_4(t/dt2)=0;
    
    
    output1(t/dt-1)=0;
    output2(t/dt-1)=0;
    
    U1_rest=0;
    U2_rest=0;
    U3_rest=0;
    U4_rest=0;
    
    gna = 120; gk = 36; gl = 0.3;
    
    hh1(t/dt,4)=0;%u,m,n,h
    hh2(t/dt,4)=0;
    
    
    % INITIALIZE STATE VARIABLES
    lif1_1(1)=0;
    lif1_2(1)=0;
    lif1_3(1)=0;
    lif1_4(1)=0;
    
    lif2_1(1)=0;
    lif2_2(1)=0;
    lif2_3(1)=0;
    lif2_4(1)=0;
    
    hh1(1,1)=0;
    hh1(1,2)=0.0489;
    hh1(1,3)=0.6446;
    hh1(1,4)=0.1341;
    hh2(1,1)=0;
    hh2(1,2)=0.0489;
    hh2(1,3)=0.6446;
    hh2(1,4)=0.1341;
    
    f=8.5;
    %x1=@(t) 9+0*t;%const
    %x1=@(t) 11.5-19*sin(1.2*t);%sine
    x1=@(t) 9-15.5*(mod(floor(t),f)-(f-1)/2-0.001)./abs(mod(floor(t),f)-(f-1)/2-0.001);%square
    %x1=@(t) 6.5+11/f*abs(t-floor(t/f)*f+f/2);%sawtooth
    %x1=@(t) 7+7/f*abs(t-floor(t/f)*f-f/2);%triangle
    x2=@(t) 0;
    W = 1e5;
    W12=W*dt1;%weight
    C=1;
    thresh=70;
    
    dt=dt1;
    for i=2:(t/dt)+1
        [hh1(i,1),hh1(i,2),hh1(i,3),hh1(i,4)]=hh_model(hh1(i-1,1),hh1(i-1,2),hh1(i-1,3),hh1(i-1,4),x1((i-1)*dt));
        if(hh1(i,1)>thresh)
            spike_hh=1;
        else
            spike_hh=0;
        end
        output_hh_1(i-1) =spike_hh;
        
        [hh2(i,1),hh2(i,2),hh2(i,3),hh2(i,4)]=hh_model(hh2(i-1,1),hh2(i-1,2),hh2(i-1,3),hh2(i-1,4),spike_hh*W12+x2((i-1)*dt));
        if(hh2(i,1)>thresh)
            output1(i-1)=1;
        else
            output1(i-1)=0;
        end
    end
    W12=W*dt2;
    dt=dt2;
    for i=2:(t/dt)+1
        x=integral(x1,(i-2)*dt,(i-1)*dt)/dt;
        lif1_1(i)=lif_simplified(C,lif1_1(i-1),U1_rest,x);%C,u,U_rest,I
        lif1_2(i)=lif_simplified(C,lif1_2(i-1),U2_rest,x);
        lif1_3(i)=lif_simplified(C,lif1_3(i-1),U3_rest,x);
        lif1_4(i)=lif_simplified(C,lif1_4(i-1),U4_rest,lif1_1(i)+lif1_2(i)+lif1_3(i));
        if(lif1_1(i)>thresh)
            spike_lif1=1;
        else
            spike_lif1=0;
        end
        if(lif1_2(i)>thresh)
            spike_lif2=1;
        else
            spike_lif2=0;
        end
        if(lif1_3(i)>thresh)
            spike_lif3=1;
        else
            spike_lif3=0;
        end
        if(lif1_4(i)>thresh)
            spike_lif4=1;
        else
            spike_lif4=0;
        end
        if(lif1_1(i)>thresh && lif1_2(i)>thresh && lif1_3(i)>thresh && lif1_4(i)>thresh)
            spike_lif=1;
            lif1_1(i)=U1_rest;
            lif1_2(i)=U2_rest;
            lif1_3(i)=U3_rest;
            lif1_4(i)=U4_rest;
        else
            spike_lif=0;
        end
        output_lif_1((i-2)*dt2/dt1+1:(i-1)*dt2/dt1)=ones(1,dt2/dt1)*spike_lif;
        lif2_1(i)=lif_simplified(C,lif2_1(i-1),U1_rest,(x2((i-1)*dt)+W12*(spike_lif1)+spike_lif4));
        lif2_2(i)=lif_simplified(C,lif2_2(i-1),U2_rest,(x2((i-1)*dt)+W12*(spike_lif2)+spike_lif4));
        lif2_3(i)=lif_simplified(C,lif2_3(i-1),U3_rest,(x2((i-1)*dt)+W12*(spike_lif3)+spike_lif4));
        lif2_4(i)=lif_simplified(C,lif2_4(i-1),U4_rest,lif2_1(i)+lif2_2(i)+lif2_3(i));
        if(lif2_1(i)>thresh && lif2_2(i)>thresh && lif2_3(i)>thresh && lif2_4(i)>thresh)
            output2((i-2)*dt2/dt1+1:(i-1)*dt2/dt1)=ones(1,dt2/dt1);
            lif2_1(i)=U1_rest;
            lif2_2(i)=U2_rest;
            lif2_3(i)=U3_rest;
            lif2_4(i)=U4_rest;
        else
            output2((i-2)*dt2/dt1+1:(i-1)*dt2/dt1)=zeros(1,dt2/dt1);
        end
    end
    spike_gap_lif=[];
    spike_gap_hh=[];
    time_lif=0;
    time_hh=0;
    for i=1:t/0.001-1
        if(output1(i+1)-output1(i)>0)
            spike_gap_hh=[spike_gap_hh,i-time_hh];
            time_hh=i;
        end
        if(output2(i+1)-output2(i)>0)
            spike_gap_lif=[spike_gap_lif,i-time_lif];
            time_lif=i;
        end
    end
    if(length(spike_gap_lif)==length(spike_gap_hh))
        disp(((sum(spike_gap_lif)-spike_gap_lif(1))-(sum(spike_gap_hh)-spike_gap_hh(1)))/(length(spike_gap_lif)-1))
    end
    
    t_d=(sum(spike_gap_hh)-spike_gap_hh(1))/(length(spike_gap_lif)-1);
    spike_gap_lif=[];
    spike_gap_hh=[];
    for i=1:t/0.001-1
        if(output1(i+1)-output1(i)>0)
            spike_gap_hh=[spike_gap_hh,i];
        end
        if(output2(i+1)-output2(i)>0)
            spike_gap_lif=[spike_gap_lif,i];
        end
    end
    if(length(spike_gap_lif)==length(spike_gap_hh))
        disp(sum(abs(spike_gap_lif-spike_gap_hh))/(length(spike_gap_lif))/t_d)
    end
    dt=dt1;
    t_current=dt*2000:dt*1000:t*1000+1*1000*dt;
    subplot(2,2,1);
    plot(t_current,output1);
    xlabel('time (ms)');
    ylabel('spike');
    hold on;
    legend("hh")
    subplot(2,2,2);
    plot(t_current,output2);
    legend("lif")
    subplot(2,2,3);
    plot(t_current,output_hh_1);
    legend("hh1")
    subplot(2,2,4);
    plot(t_current,output_lif_1);
    legend("lif1")
end