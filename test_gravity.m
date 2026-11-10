[V,F] = load_mesh('~/Dropbox/models/cartoon-elephant_fixed.obj');
name = 'elephant';
V = V*axisangle2matrix([0 1 0],-pi/2);
V = V*axisangle2matrix([1 0 0],-pi/2);
V = V*axisangle2matrix([0 0 1],pi);
V = V-min(V);
V = V./max(abs(V(:)));
[dV,dF] = decimate_libigl(V,F,2000,'Method','progressive-hulls');
%name = 'ramp';
%rh = 0.9;
%V = [0 0 rh;0 1 rh;1 1 rh;1 0 rh;1 0 1;1 1 1];
%F = convhull(V);
%[dV,dF] = deal(V,F);

rng(0);
np = 1000;
d_hat = 0.01;
P = [randn(np,2).*0.1+mean(V(:,1:2))];
P(:,3) = max(V(:,3)) + d_hat*10 + d_hat*randn(np,1);
%P = rand(np,2);
%P(:,3) = 1.1;

gravity = [0 0 -9.81];

tc = [1.0 0.8 0.6];
tsh = tsurf(F,V,'FaceVertexCData',repmat(tc,size(V,1),1),'EdgeColor','none',fsoft,falpha(1,0),fphong);
hold on;
bc = [0.4 0.2 0.1];
PC = okloop(size(P,1));
ssh = sct(P,'filled','CData',PC,'MarkerEdgeColor',bc*0.5,'SizeData',20,'LineWidth',0.5);
hold off;
axis equal;
view(-20,56);
%apply_ambient_occlusion(tsh,'SoftLighting',false,'AddLights',false);
l = add_lights;
%view(-90,0);
view(3);
shadow_nudge = 1e-2;
hsh = add_shadow({tsh,ssh},l{5},'Nudge',shadow_nudge);
axis manual;

nsubsteps = 10;
dt = 1/30 / nsubsteps;
t_max = 3;
t = 0;
P_dot = zeros(size(P));

% Time integrator: 'backward-euler' or 'newmark'
%
% Both are written as an incremental potential (cf. C-IPC eq. for generic
% integrators):
%
%   E(p) = ½‖p - p̂‖²_M + c·B(p)
%
% where the predictor p̂ and the barrier coefficient c are integrator
% specific.
%integrator = 'backward-euler';
integrator = 'newmark';
% Newmark parameters (¼,½) ↦ average acceleration / trapezoidal: no
% algorithmic damping (so impacts actually rebound).
%beta_N  =  0.25;
%gamma_N = 0.5;

% less extreme bounces...
beta_N  =  1;
gamma_N = 0.7;
beta_N  =  1;
gamma_N = 1;

% Acceleration state (only used by Newmark). At t=0 the particles are in
% free fall (no contact), so a₀ = g.
P_ddot = repmat(gravity,size(P,1),1);

kappa = 1;
max_Z = max(P(:,3));
substep_iter = 1;
P_trail = [];
while  t<t_max
  t = t + dt;
  switch integrator
  case 'backward-euler'
    % p̂ = pⁿ + h vⁿ + h² M⁻¹ f_ext,   E(p) = ½‖p-p̂‖² + h² B(p)
    P_hat = P + dt*P_dot + dt^2 * gravity;
    c = dt^2;
  case 'newmark'
    % p̂ = pⁿ + h vⁿ + h²(½-β) aⁿ + β h² g,   E(p) = ½‖p-p̂‖² + β h² B(p)
    P_hat = P + dt*P_dot + dt^2*(0.5-beta_N)*P_ddot + beta_N*dt^2*gravity;
    c = beta_N*dt^2;
  otherwise
    error('Unknown integrator: %s',integrator);
  end
  P_next = nan(size(P));
  for i = 1:size(P,1)
    max_inner = 60;
    p = P(i,:);
    p_hat = P_hat(i,:);

      b_fun = @(p) barrier(@(p) min_distance(p,@(p) distance_to_mesh(p,dV,dF),@(p) distance_to_ground(p)),p,d_hat);
      [~,dbdp] = b_fun(p);

    g0 = p - p_hat;
    gc = c * dbdp;
    kappa_g = -dot(gc,g0) / dot(gc,gc);
    kappa_g = min(max(kappa_g,1e-5),1e5);

    for inner = 1:max_inner
      % argmin_p ½ ‖ p - p̂ ‖² + c b(d(p))
      %b_fun = @(p) barrier(@distance_to_ground,p,d_hat);
      %b_fun = @(p) barrier(@(p) distance_to_mesh(p,dV,dF),p,d_hat);
      f = @(p) 0.5*sum((p - p_hat).^2) + c * kappa_g * b_fun(p);
      [b,dbdp,d2bdp2] = b_fun(p);
      g = (p - p_hat) + c * kappa_g * dbdp;
      H = eye(3) + c * kappa_g * d2bdp2;
      %H = reshape(psd_project_rows(reshape(H,1,[])),3,3);
      dp = -(pinv(H)*g')';
      %dp = -g;
      if norm(g) < 1e-6
        break;
      end
      %for pass = 1:2
        [ss,p] = backtracking_line_search(f,p,g,dp,0.3,0.5,60);
        if ss == 0 
          fprintf('%d at %d: |g| = %g, |dp| = %g (line search failed)\n',i,inner,norm(g),norm(dp));
          break;
        end
      %  if ss == 0
      %    if pass == 1
      %      dp = -g;
      %    else
      %      warning('Line search failed, even after trying to use gradient');
      %      break;
      %    end
      %  else
      %    break;
      %  end
      %end
    end
    P_next(i,:) = p;
  end

  switch integrator
  case 'backward-euler'
    P_dot = (P_next - P)/dt;
  case 'newmark'
    % Invert the Newmark position update for aⁿ⁺¹, then the velocity update.
    P_ddot_next = (P_next - P - dt*P_dot - dt^2*(0.5-beta_N)*P_ddot)/(beta_N*dt^2);
    P_dot = P_dot + dt*((1-gamma_N)*P_ddot + gamma_N*P_ddot_next);
    P_ddot = P_ddot_next;
  end
  P = P_next;

  if mod(substep_iter,nsubsteps) == 1
    cellfun(@(h) delete(h),hsh);
    hsh = add_shadow({tsh,ssh},l{5},'Nudge',shadow_nudge);
    set(ssh,'XData',P(:,1),'YData',P(:,2),'ZData',P(:,3));
    drawnow;
    figgif(sprintf('%s-%d-%s.gif',name,np,integrator));
    P_trail = cat(3,P_trail,P);
  end
  max_Z = [max_Z;max(P(:,3))];
  substep_iter = substep_iter + 1;
end
%plot(max_Z);


function [b,dbdp,d2bdp2] = barrier(d_fun,p,d_hat)
  % b(d,d̂) = { -(d-d̂)² log(d/d̂) , 0 < d < d̂ ; 0 , d ≥ d̂ }
  switch nargout
    case 1
      d = d_fun(p);
    case 2
      [d,dddp] = d_fun(p);
    case 3
      [d,dddp,d2ddp2] = d_fun(p);
  end
  if d <= 0
    assert(nargout == 1);
    b = inf;
  elseif d > d_hat
    b = 0;
    if nargout > 1
      dbdp = zeros(size(p));
    end
    if nargout > 2
      d2bdp2 = zeros(length(p),length(p));
    end
  else
    assert(d > 0,'Barrier evaluated with derivatives at non-positive distance');
    b = -(d-d_hat)^2 * log(d/d_hat);
    if nargout > 1
      dbdd = -2*(d-d_hat)*log(d/d_hat) - (d-d_hat)^2/d;
      dbdp = dbdd * dddp;
    end
    if nargout > 2
      d2bdd = -2*log(d/d_hat) - 4*(d-d_hat)/d + (d-d_hat)^2/d^2;
      d2bdp2 = d2bdd * (dddp' * dddp) + dbdd * d2ddp2;
    end
  end

end

function [f,g,H] = min_distance(p,A,B)
  switch nargout
    case 1
      f = min(A(p),B(p));
    case 2
      [fa,dfa] = A(p);
      [fb,dfb] = B(p);
      if fa < fb
        f = fa;
        g = dfa;
      else
        f = fb;
        g = dfb;
      end
    case 3
      [fa,dfa,Hfa] = A(p);
      [fb,dfb,Hfb] = B(p);
      if fa < fb
        f = fa;
        g = dfa;
        H = Hfa;
      else
        f = fb;
        g = dfb;
        H = Hfb;
      end
  end
end

function [f,dfdp,d2fdp2] = distance_to_mesh(p,V,F)
  [f,I,C] = signed_distance(p,V,F,'SignedDistanceType','fwn');
  if nargout > 1
    dfdp = (p - C)./f;
  end
  tol = 1e-10;
  if nargout > 2
    B = barycentric_coordinates(C,V(F(I,1),:),V(F(I,2),:),V(F(I,3),:));
    on_face = sum(B<tol,2)==0;
    on_edge = sum(B<tol,2)==1;
    on_vertex = sum(B<tol,2)==2;
    if on_face
      d2fdp2 = zeros(3,3);
    elseif on_vertex
      d2fdp2 = eye(3)/f - (p - C)'*(p - C)/f^3;
    else
      % on edge
      v1 = V(F(I,1),:);
      v2 = V(F(I,2),:);
      v3 = V(F(I,3),:);

      if B(1) < tol
        v1 = v2;
        v2 = v3;
      elseif B(2) < tol
        v2 = v3;
      end

      e = (v2-v1)/norm(v2-v1);

      r = p-C;
      d2fdp2 = (eye(3)-e'*e)/f - r'*r/f^3;
    end
  end
end

function [z,dzdp,d2zdp2] = distance_to_ground(p)
  z = p(3);
  if nargout > 1
    dzdp = [0 0 1];
  end
  if nargout > 2
    d2zdp2 = zeros(3,3);
  end
end

