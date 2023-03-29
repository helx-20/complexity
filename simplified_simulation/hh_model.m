function [new_u,new_m,new_n,new_h]=hh_model(u,m,n,h,I)
    global dt;
    global gk;
    global gna;
    global gl;
    
    % Nernst Potentials
    Ena =115; Ek = -12; El = 10.613;

    % Maximum Conductances
    gna1 = 120; gk1 = 36; gl1 = 0.3;

    % Membrane Capacitance
    C = 1;

    an = @(u) 0.01*((-10+u)/(1-exp(-(-10+u)/10)));
    am = @(u) 0.1*((-25+u)/(1-exp(-(-25+u)/10)));
    ah = @(u) 0.07*exp(-(u)/20);
    bn = @(u) 0.125*exp(-(u)/80);
    bm = @(u) 4*exp(-(u)/18);
    bh = @(u) 1/(1+exp(-(-30+u)/10));

    % solve for membrane currents
    Ik  = gk1  * n^4      * (u - Ek);
    Ina = gna1 * m^3 * h  * (u - Ena);
    Il  = gl1  *            (u - El);   
    Imem = Ik + Ina + Il;

    
    gk=gk1  * n^4;
    gna=gna1 * m^3 * h;
    gl=gl1;

    % define the state variable derivatives
    du = - Imem/C + I/C;
    dm = am(u) * (1-m) - bm(u) * m;
    dh = ah(u) * (1-h) - bh(u) * h;
    dn = an(u) * (1-n) - bn(u) * n;

    % use forward euler to increment the state variables
    new_u = u + du*dt;
    new_m = m + dm*dt;
    new_h = h + dh*dt;
    new_n = n + dn*dt;

end