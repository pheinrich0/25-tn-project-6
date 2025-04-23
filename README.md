# Julia Code Repository for the 2025 Tensor Networks Lecture

This is the git repository for julia version of the [2025 tensor networks lecture at LMU Munich](https://moodle.lmu.de/course/view.php?id=40399).

## How to get this repository

1. Install git. You can get it from [https://git-scm.com/] for (almost) all operating systems.
2. Open a terminal. (On windows, this has to be the git terminal you just installed.)
3. Navigate to a folder in which you want your files for this course.
3. Copy-paste this into the terminal and press enter:
```bash
git clone https://gitlab.physik.uni-muenchen.de/25tn/25tn-julia.git
```
4. A new folder called `25tn-julia` will be created with all material contained in this repository.

## How to update the repository during the semester

1. Open a terminal in the `25tn-julia` folder.
2. Type the following into the terminal and press enter:
```bash
git pull
```

## How to setup Julia for the course

To solve the problem sets, you'll have to install Julia. Download and follow the installation instructions for your operating system provided at
[https://julialang.org/].



### Test that you can execute Julia.
- On Linux and MacOS:
    
    Open a terminal in the `25tn-julia` folder, and type
    ```bash
    julia
    ```

- On Windows:
    
    Open the julia application you installed earlier. If you used default settings, it is available in the start menu; if you customized anything, you already know where to find it.

When starting julia, like this should appear in the terminal window:
```
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.11.5 (2025-04-14)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia>
```

The version number (in my case 1.11.5) might be different. Any version >= 1.10 should be OK; if you have version 1.9 or earlier, please update your installation.

Now, type `]` into the terminal. The `julia>` prompt will switch to
```
(@v1.11) pkg>
```
or similar. You are now in "Pkg mode", which is used to download, install and update julia packages. You can get out of Pkg mode with `backspace`.

In package mode, type the following:
```
(@v1.11) pkg> activate /path/25tn-julia
```
and replace `/path/` with the path to the parent folder you put `25tn-julia` into. The *prompt* (the text to the left of your cursor) should switch to:
```
(tn_julia) pkg>
```
which means you now activated the project environment for this course. This is important to ensure that everyone has the same packages installed on their system. Now, make julia download and install those packages you will need:
```
(tn_julia) pkg> instantiate
```
After typing this command and pressing enter, you should see a number of packages being downloaded and installed. This might take a while - fortunately, you only have to go through this once. Once everything has been downloaded and installed, you're ready to go!

To test this, try to execute the file `problem/example/problem1.jl`:
```julia
julia> include("problem/example/problem1.jl")
```
You should see a plot that shows a gaussian function.

## VS Code

I recommend using VS code as your editor. Install the julia extension, and it becomes a powerful integrated development environment. Executing, testing and debugging are made much easier with it. There is a nice guide here: https://code.visualstudio.com/docs/languages/julia

## How to structure your code

All solutions to problem sets should go into the `problem` directory; for example, there is some pre-written code for problem set 01 in `problem/set01`. Generally, it pays off to write at least 3 files:
1. **Function definitions**:

    A file that defines functions that solve specific parts of the problem. For more difficult tasks, grouping function definitions into multiple files (by conceptual relation) might be useful. For an example, see `problem/example/gaussian.jl`.

2. **Unit tests**:
    
    One file has tests for the functions you defined above. This can be split into multiple files as well. For example, see `problem/example/tests.jl`.

3. **Script**:
    
    One file is an executable script that calls your functions and assembles them into a solution of the whole problem. This file might also generate plots or create data files. For an example, see `problem/example/problem1.jl`.

This structure is useful, because it allows using  the function definitions in future problem sets!

Some hints about each file:

### Function definitions file
Your function definitions should be part of the package `tn_julia`, which is defined by the file `src/tn_julia.jl`:
```julia
module tn_julia

# Write your package code here.

end
```
In theory, you could write your code directly into this file, but that is *generally a bad idea*: The file will quickly grow to contain multiple thousand lines and become unreadable. Instead, separate your code into small files that each supply one specific set of function definitions. For example, you might have a file `src/gaussian.jl` that contains a gaussian function. Then, include that file in `src/tn_julia.jl` as follows:
```julia
module tn_julia

include("gaussian.jl")
# ... other includes ...

end

```

If you need functionality from other packages, such as `LinearAlgebra` or `JLD2`, you have to import them at the top of your file, like this:
```julia
using LinearAlgebra
using JLD2
```
If the package needs these in general, it's a good idea to put this into the main package file:
```julia
module tn_julia

using LinearAlgebra
using JLD2

include("gaussian.jl")
# ... other includes ...

end
```

### Unit tests file
To perform unit tests, you need to import `Test`, and the file that defines those functions you're going to test. Similar to the contents of `src/`, we will subdivide the tests into multiple files, where each file tests a particular set of features, and have a "main" file `runtests.jl`, which includes all others. For example:
```julia
using tn_julia
using Test

include("test_gaussian.jl")
```
The file `test_gaussian.jl` contains the following:
```julia
import tn_julia: gaussian

@testset "Basic properties of a gaussian" begin
    @test gaussian(0.0) == 1
    @test gaussian(1.0) == exp(-1.0)
    for x in 0:0.1:1
        @test gaussian(x) == gaussian(-x)
    end
end

```
Group your `@test`s into multiple `@testset`s to test multiple sets of functions. More details will be shown during the tutorial.

### Script
Your script file has to import your function definitions, and potentially additional libraries for plotting, saving data, etc. For example:
```julia
using Plots
using LaTeXStrings

import tn_julia: gaussian

xvals = -3:0.1:3
plot(xvals, gaussian.(xvals), xlabel=L"x", ylabel=L"g(x)")
```
Generally, most non-trivial code is useful in more than one place. Therefore, you should get into the habit of writing the non-trivial parts as functions, and then assemble those functions to a complete program. That way, your function can be re-used in a different place, and you don't end up solving the same problems over and over again.

## How to make plots

For plots, it's generally easiest to use the `Plots` library. To plot a quantity `y` against parameter `x`, simply type
```julia
using Plots
plot(x, y)
```
and a plot will appear. To get Latex plot labels, import the `LaTeXStrings` library and prepend your latex labels with `L`, like this:
```julia
using LaTeXStrings
plot(x, y, L"\gamma", L"e^{-\gamma^2}")
```

## How to hand in solutions
Put your code in the `problem/setXX/` directory. Make sure it can be executed without modifications. Put the entire `23tn-julia` directory into a zip file and upload it on Moodle.
