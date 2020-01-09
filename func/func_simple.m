classdef func_simple < handle
% Define interface of simple closed convex function Psi(x)
    methods (Abstract)
        [fval, grad] = oracle(Psi, x)
        % Return function value Psi(x), and a gradient

    end
end