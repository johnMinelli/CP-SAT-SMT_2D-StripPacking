
include "globals.mzn";
include "diffn.mzn";
%include "data.dzn";


%------------------------------------------------------------------------------%
% Model parameters

    % width of bounding box (= circuit plate).
int: width;
    % number of rectangles (= circuits).
int: n_rects;
    % all rectangles.
set of int: RECTS = 1..n_rects;
array[RECTS] of int: width_rects; % widths of rectangles
array[RECTS] of int: height_rects; % heights of rectangles
    % bounds
int: min_height = max(height_rects);
int: max_height = sum(height_rects);

    % aim of the model: find the rectangles positions and the bounding box height
array[RECTS] of var 0..width : X;
array[RECTS] of var 0..max_height: Y;
var min_height..max_height: HEIGHT;

%------------------------------------------------------------------------------%
% Model representation
%
%             +------------------------------------+
%             |                                    |
% sum_rects ≥ |                                    |
%   Height    |                                    |
% ≥ max_rect  |                                    |
%             |                                    |
%             +------------------------------------+
%                         Width (fixed)
%
%------------------------------------------------------------------------------%
% Contraints
constraint
    forall( i, j in RECTS where i < j ) (
        X[i] + width_rects[i] <= X[j] 
    \/  X[j] + width_rects[j] <= X[i] 
    \/  Y[i] + height_rects[i] <= Y[j]
    \/  Y[j] + height_rects[j] <= Y[i]
    );
        

constraint
forall(rect in RECTS)(
X[rect] + width_rects[rect] <= width /\
Y[rect] + height_rects[rect] <= HEIGHT
);


%
%------------------------------------------------------------------------------%



% Objective
solve::  
int_search([ HEIGHT ]
            ++ [ X[i] | i in RECTS ]
            ++ [ Y[i] | i in RECTS ],
            input_order, indomain_min, complete
            ) minimize HEIGHT;
