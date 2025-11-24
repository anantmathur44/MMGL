# Numerical Experiments for "A Fast MM Algorthm for Group Lassp" – README

This repository contains Julia scripts used to generate the simulation and data analysis results in the paper "A Fast MM Algorthm for Group Lasso". The output files are produced in CSV format. The scripts are organized into three main folders:
- **Simulations**
- **Real Data**
- **Pseudo Real Data**

Each CSV output file is created in the same directory as its corresponding script, and the file names include simulation-specific parameters (such as simulation index, parameter values, etc.) to distinguish different runs.

---

## 1. Simulations

### 1.1 Paths Simulation
- **Script Location:**  
  `Simulations/sim_conv_paths/PathsSim.jl`
- **Output File:**  
  `path_<i>.csv`  
  *(Here, `<i>` is the simulation index.)*
- **Output Columns:**
  1. **Cumulative MM Iterations:** Running total of iterations performed by the MM algorithm.
  2. **Cumulative BCD Iterations:** Running total of iterations performed by the block coordinate descent algorithm.
  3. **Cumulative MM Time:** Running total of computation time for the MM algorithm.
  4. **Cumulative BCD Time:** Running total of computation time for the BCD algorithm.
  5. **Regularization Parameter Values:** The lambda values used in the regularization path.
  6. **Test Error:** The measured error on the test set for each lambda value.

---

### 1.2 J vs. Time Simulation
- **Script Location:**  
  `Simulations/sim_m_vs_time/m_vs_time.jl`
- **Output File:**  
  `m_time_<m>_<i>.csv`  
  *(Here, `<m>` indicates the number of groups and `<i>` is the simulation index.)*
- **Output Columns:**
  1. **Cumulative MM Iterations**
  2. **Cumulative BCD Iterations**
  3. **Cumulative MM Time**
  4. **Cumulative BCD Time**

---

### 1.3 N vs. Time Simulation
- **Script Location:**  
  `Simulations/sim_n_vs_time/sim_n_vs_time.jl`
- **Output File:**  
  `n_time_<n>_<i>.csv`  
  *(Here, `<n>` indicates number of responses, and `<i>` is the simulation index.)*
- **Output Columns:**
  1. **Cumulative MM Iterations**
  2. **Cumulative BCD Iterations**
  3. **Cumulative MM Time**
  4. **Cumulative BCD Time**

---

### 1.4 Simulation Settings
These scripts simulate different settings as discussed in the Numerical Simulations section of the paper. The CSV files are named using specific simulation parameters (e.g., values for rho and psi) and the simulation index.

#### Scripts:
- `Simulations/sim_setting1_d3/snr_1_setting1_grp3.jl`
- `Simulations/sim_setting1_d5/snr_1_setting1_grp5.jl`
- `Simulations/sim_setting2_d3/snr_1_setting2_grp3.jl`
- `Simulations/sim_setting2_d5/snr_1_setting2_d5.jl`

**Output File Format:**  
`rho_<rho>_psi_<psi>_sim_<i>.csv`  
*(Here, `<rho>`, `<psi>`, and `<i>` represent the correlation structure and index.)*

**Output Columns:**
1. **MM Time:** Computation time for the MM algorithm.
2. **BCD Time:** Computation time for the BCD algorithm.
3. **MM Iterations:** Number of iterations performed by the MM algorithm.
4. **BCD Iterations:** Number of iterations performed by the BCD algorithm.

---

## 2. Real Data

### 2.1 Real Data Analysis
- **Script Location:**  
  `Real Data/real_data.jl`
- **Output Files:**
  - **`real_data.csv`**  
    **Output Columns:**
    1. **Cumulative MM Time:** Running total of the MM algorithm’s computation time.
    2. **Cumulative BCD Time:** Running total of the BCD algorithm’s computation time.
    3. **Cumulative MM Iterations:** Running total of iterations for the MM algorithm.
    4. **Cumulative BCD Iterations:** Running total of iterations for the BCD algorithm.
    5. **Regularization Parameter Values:** The lambda values used for the regularization path.
    
  - **`test_error.csv`**  
    **Output Columns:**
    1. **BCD Test Error:** Test error associated with the BCD algorithm.
    2. **MM Test Error:** Test error associated with the MM algorithm.

---

## 3. Pseudo Real Data

These scripts output results for simulations on pseudo real data under different settings.

#### Scripts:
- `Pseudo Real Data/mice_setting1_d10.jl`
- `Pseudo Real Data/mice_setting1_d25.jl`
- `Pseudo Real Data/mice_setting2_d10.jl`
- `Pseudo Real Data/mice_setting2_d25.jl`

**Output File Formats:**
- For `mice_setting1_d10.jl`: `Mice_Setting1_Grp10_sim_<i>.csv`
- For `mice_setting1_d25.jl`: `Mice_Setting1_Grp25_sim_<i>.csv`
- For `mice_setting2_d10.jl`: `Mice_Setting2_Grp10_sim_<i>.csv`
- For `mice_setting2_d25.jl`: `Mice_Setting2_Grp25_sim_<i>.csv`  
  *(Here, `<i>` denotes the simulation index.)*

**Output Columns:**
1. **MM Time:** Computation time for the MM algorithm.
2. **CD Time:** Computation time for the comparison (CD) algorithm.
3. **MM Iterations:** Number of iterations performed by the MM algorithm.
4. **CD Iterations:** Number of iterations performed by the comparison algorithm.

---

## Execution

Runing run_all.jl will traverse and run all experiment scripts.

Alternatively, each script can be run individually. When executed, the scripts create their respective CSV output files in the same directories as the scripts.

Please refer to this README for an overview of the outputs and to help interpret the CSV files generated by the simulations.

---



