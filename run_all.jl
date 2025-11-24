#!/usr/bin/env julia

# List of script paths relative to the submission folder.
scripts = [
    "Simulations/sim_conv_paths/PathsSim.jl",
    "Simulations/sim_m_vs_time/m_vs_time.jl",
    "Simulations/sim_n_vs_time/sim_n_vs_time.jl",
    "Simulations/sim_setting1_d3/snr_1_setting1_grp3.jl",
    "Simulations/sim_setting1_d5/snr_1_setting1_grp5.jl",
    "Simulations/sim_setting2_d3/snr_1_setting2_grp3.jl",
    "Simulations/sim_setting2_d5/snr_1_setting2_d5.jl",
    "Real Data/real_data.jl",
    "Pseudo Real Data/mice_setting1_d10.jl",
    "Pseudo Real Data/mice_setting1_d25.jl",
    "Pseudo Real Data/mice_setting2_d10.jl",
    "Pseudo Real Data/mice_setting2_d25.jl"
]

# Iterate over each script path.
for script_rel in scripts
    # Compute the full path relative to the current working directory (submission folder).
    full_script_path = joinpath(pwd(), script_rel)
    # Extract the directory in which the script resides.
    script_dir = dirname(full_script_path)
    # Extract the script filename.
    script_file = basename(full_script_path)
    
    println("---------------------------------------------------------")
    println("Processing: ", script_rel)
    
    if !isfile(full_script_path)
        println("Warning: File does not exist: ", full_script_path)
        continue
    end
    
    # Change into the directory containing the script and run it.
    cd(script_dir) do
        println("Current directory: ", pwd())
        println("Running script: ", script_file)
        try
            run(`julia $script_file`)
        catch err
            println("Error running ", script_file, ": ", err)
        end
    end
end

println("---------------------------------------------------------")
println("All scripts processed.")
