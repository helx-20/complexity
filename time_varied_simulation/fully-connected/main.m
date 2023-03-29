function main()%fc
    close all;
    global dt;
    global gk;
    global gna;
    global gl;
    global C;
    dt=0.001;%s
    t=40;%s
    cfg_fc=[128,10];
    lif1_1(cfg_fc(1),t/dt)=0;
    lif1_2(cfg_fc(1),t/dt)=0;
    lif1_3(cfg_fc(1),t/dt)=0;
    lif1_4(cfg_fc(1),t/dt)=0;
    
    lif2_1(cfg_fc(2),t/dt)=0;
    lif2_2(cfg_fc(2),t/dt)=0;
    lif2_3(cfg_fc(2),t/dt)=0;
    lif2_4(cfg_fc(2),t/dt)=0;
    
    output1(cfg_fc(2),t/dt-1)=0;
    output2(cfg_fc(2),t/dt-1)=0;
    
    U1_rest=50;
    U2_rest=-77;
    U3_rest=-54.387;
    U4_rest=0;
    
    gna = 120; gk = 36; gl = 0.3;
    gs=gna+gk+gl;
    
    hh1(cfg_fc(1),t/dt,4)=0;%u,m,n,h
    hh2(cfg_fc(2),t/dt,4)=0;
    
    % INITIALIZE STATE VARIABLES
    lif1_1(:,1)=zeros(cfg_fc(1),1);
    lif1_2(:,1)=zeros(cfg_fc(1),1);
    lif1_3(:,1)=zeros(cfg_fc(1),1);
    lif1_4(:,1)=-70*ones(cfg_fc(1),1);
    f1=zeros(cfg_fc(1),1);
    f2=zeros(cfg_fc(1),1);
    f3=zeros(cfg_fc(1),1);
    
    lif2_1(:,1)=zeros(cfg_fc(2),1);
    lif2_2(:,1)=zeros(cfg_fc(2),1);
    lif2_3(:,1)=zeros(cfg_fc(2),1);
    lif2_4(:,1)=-70*ones(cfg_fc(2),1);
    f4=zeros(cfg_fc(2),1);
    f5=zeros(cfg_fc(2),1);
    f6=zeros(cfg_fc(2),1);
    
    hh1(:,1,1)=-70*ones(cfg_fc(1),1);
    hh1(:,1,2)=0.0289*ones(cfg_fc(1),1);
    hh1(:,1,3)=0.2446*ones(cfg_fc(1),1);
    hh1(:,1,4)=0.7541*ones(cfg_fc(1),1);
    hh2(:,1,1)=-70*ones(cfg_fc(2),1);
    hh2(:,1,2)=0.0289*ones(cfg_fc(2),1);
    hh2(:,1,3)=0.2446*ones(cfg_fc(2),1);
    hh2(:,1,4)=0.7541*ones(cfg_fc(2),1);  
    
    image=imread('input.png');
    image=double(image(:));
    
    thresh=-40;
    W=load('weight.mat');
    W01=W.W01;
    W12=W.W12;
    C=1;
    input=W01'*image;
    
    for i=2:(t/dt)
        %hh_1
        [hh1(:,i,1),hh1(:,i,2),hh1(:,i,3),hh1(:,i,4)]=hh_model_multiple(hh1(:,i-1,1),hh1(:,i-1,2),hh1(:,i-1,3),hh1(:,i-1,4),input);
        spike_hh1=(hh1(:,i-1,1)>thresh);
        gs=gna+gk+gl;
        lif1_1(:,i)=lif_model_multiple(C,lif1_1(:,i-1),U1_rest,input./gs);%C,u,U_rest,I
        lif1_2(:,i)=lif_model_multiple(C,lif1_2(:,i-1),U2_rest,input./gs);
        lif1_3(:,i)=lif_model_multiple(C,lif1_3(:,i-1),U3_rest,input./gs);
        f1=f1+(lif1_1(:,i)-lif1_1(:,i-1)).*gna;
        f2=f2+(lif1_2(:,i)-lif1_2(:,i-1)).*gk;
        f3=f3+(lif1_3(:,i)-lif1_3(:,i-1)).*gl;
        lif1_4(:,i)=lif_model_multiple(C./gs,lif1_4(:,i-1),U4_rest,-(f1+f2+f3)+(gna.*lif1_1(:,i-1)+gk.*lif1_2(:,i-1)+gl.*lif1_3(:,i-1))./gs);
        spike_lif1=(f1+f2+f3+lif1_4(:,i-1)>thresh);
        %hh_2
        [hh2(:,i,1),hh2(:,i,2),hh2(:,i,3),hh2(:,i,4)]=hh_model_multiple(hh2(:,i-1,1),hh2(:,i-1,2),hh2(:,i-1,3),hh2(:,i-1,4),W12'*spike_hh1);
        output1(:,i-1)=(hh2(:,i-1,1)>thresh);
        %lif_hh_2
        gs=gna+gk+gl;
        lif2_1(:,i)=lif_model_multiple(C,lif2_1(:,i-1),U1_rest,W12'*spike_lif1./gs);
        lif2_2(:,i)=lif_model_multiple(C,lif2_2(:,i-1),U2_rest,W12'*spike_lif1./gs);
        lif2_3(:,i)=lif_model_multiple(C,lif2_3(:,i-1),U3_rest,W12'*spike_lif1./gs);
        f4=f4+(lif2_1(:,i)-lif2_1(:,i-1)).*gna;
        f5=f5+(lif2_2(:,i)-lif2_2(:,i-1)).*gk;
        f6=f6+(lif2_3(:,i)-lif2_3(:,i-1)).*gl;
        lif2_4(:,i)=lif_model_multiple(C./gs,lif2_4(:,i-1),U4_rest,-(f4+f5+f6)+(gna.*lif2_1(:,i-1)+gk.*lif2_2(:,i-1)+gl.*lif2_3(:,i-1))./gs);
        output2(:,i-1)=(f4+f5+f6+lif2_4(:,i-1)>thresh);
    end
    save("data.mat",'output1','output2')
    t_current=dt*2000:dt*1000:t*1000;
    for i=1:cfg_fc(2)
        subplot(cfg_fc(2),1,i)
        plot(t_current,output1(i,:));
        hold on;
        plot(t_current,output2(i,:));
        if(i==1)
            legend("hh","lif-hh")
        end
        if(i==cfg_fc(2)/2)
            ylabel('spike');
        end
        if(i==cfg_fc(2))
            xlabel('time (ms)');
        end
        set(gca, 'yticklabel', get(gca, 'ytick'));
        if(i~=cfg_fc(2))
            set(gca, 'xticklabel', []);
        else
            set(gca, 'xticklabel', get(gca, 'xtick'));
        end
    end
end