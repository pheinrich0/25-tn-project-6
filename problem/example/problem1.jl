using Plots
using LaTeXStrings

import tn_julia: gaussian

xvals = -3:0.1:3
plot(xvals, gaussian.(xvals), xlabel=L"x", ylabel=L"g(x)")
