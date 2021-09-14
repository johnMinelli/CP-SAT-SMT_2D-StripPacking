
include "globals.mzn";
include "diffn.mzn";

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

    % rectangles ordered by X and Y position
array[RECTS] of var int: ordered_rects;

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

% Global packing constraint: avoid overlap simultaneously on both axes
constraint diffn(X, Y, width_rects, height_rects);

constraint
forall(rect in RECTS)(
X[rect] + width_rects[rect] <= width /\
Y[rect] + height_rects[rect] <= HEIGHT
);

% redundant
constraint redundant_constraint(
  cumulative(Y, height_rects, width_rects, width) /\
  cumulative(X, width_rects, height_rects, HEIGHT)
);


% no gaps
constraint
forall(r in RECTS)(
member([0] ++ [X[s] + width_rects[s] | s in RECTS where s != r], X[r]) /\
member([0] ++ [Y[s] + height_rects[s] | s in RECTS where s != r], Y[r]));



constraint forall(i in index_set(ordered_rects) where i>1)(
    lex_lesseq([X[ordered_rects[i-1]],Y[ordered_rects[i-1]]],[X[ordered_rects[i]],Y[ordered_rects[i]]])
);


% symmetry breaking for equals rect interchangable

constraint
  forall(r in RECTS)(
    let {array[int] of int: EQ = [i | i in RECTS where width_rects[i]=width_rects[r] /\
        height_rects[i]=height_rects[r]]} in
      if length(EQ)>1 /\ min(EQ)=r then
        forall(i in index_set(EQ) where i>1)(
          lex_less([Y[EQ[i-1]],X[EQ[i-1]]],[Y[EQ[i]],X[EQ[i]]])
        )
      else true endif
  );
  
  
% symmetry breaking for polygonal subregions of rectangles interchangable
  
constraint let { var 0..n_rects-1: y_min; var 0..n_rects-1: x_min; var 1..n_rects: y_max; var 1..n_rects: x_max; array[2..n_rects-1] of var 1..n_rects: SUB} in
  if y_min>y_max /\ x_min>x_max /\ alldifferent(SUB) /\
  forall(rect in SUB)(
    X[rect] + width_rects[rect] <= x_max /\ x_min <= X[rect] /\
    Y[rect] + height_rects[rect] <= y_max /\ y_min <= Y[rect]
  ) /\
  forall(xi in  x_min..x_max)(
    exists(b in SUB)(X[b]<=xi /\ xi<=X[b]+width_rects[b] /\ y_max=Y[b]+height_rects[b]) /\
    exists(b in SUB)(X[b]<=xi /\ xi<=X[b]+width_rects[b] /\ y_min=Y[b])
  ) /\
  forall(yi in  y_min..y_max)(
    exists(b in SUB)(Y[b]<=yi /\ yi<=Y[b]+height_rects[b] /\ x_max=X[b]+width_rects[b]) /\
    exists(b in SUB)(Y[b]<=yi /\ yi<=Y[b]+height_rects[b] /\ x_min=X[b])
  ) then
    % set constraint to not allow other orderings when you have this subregion of rectangles
    forall(m,n in ordered_rects where m<n)(
      if(m in array2set(SUB) /\ n in array2set(SUB)) then
         lex_lesseq([X[m],Y[m]],[X[n],Y[n]])
       else true endif
     )
    % use solutions with rectangles set (subregion) ordered by size
    %forall(i in index_set(SUB) where i>1)(
      %    lex_greatereq([height_rects[SUB[i-1]],width_rects[SUB[i-1]]],[height_rects[SUB[i]],width_rects[SUB[i]]])
     %   )
  else true endif;
  
%
%------------------------------------------------------------------------------%



% Objective
solve::  
int_search([ HEIGHT ]
            ++ [ X[i] | i in RECTS ]
            ++ [ Y[i] | i in RECTS ],
            input_order, indomain_min, complete
            ) minimize HEIGHT;
