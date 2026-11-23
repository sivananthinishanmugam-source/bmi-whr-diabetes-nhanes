# NHANES 2017–2018 raw source files

The exact fourteen files below are required. GHB_J is deliberately not included because HbA1c is not used. Download SAS XPORT files, not HTML documentation, and save with the stated names in data/raw. Existing copies are included for this local rerun.

| File | CDC data download | CDC documentation |

|---|---|---|

| DEMO_J.xpt | [XPORT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt) | [Documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.htm) |

| BMX_J.xpt | [XPORT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.xpt) | [Documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.htm) |

| BPX_J.xpt | [XPORT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPX_J.xpt) | [Documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPX_J.htm) |

| BPQ_J.xpt | [XPORT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPQ_J.xpt) | [Documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPQ_J.htm) |

| DIQ_J.xpt | [XPORT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DIQ_J.xpt) | [Documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DIQ_J.htm) |

| GLU_J.xpt | [XPORT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GLU_J.xpt) | [Documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GLU_J.htm) |

| HDL_J.xpt | [XPORT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/HDL_J.xpt) | [Documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/HDL_J.htm) |

| TCHOL_J.xpt | [XPORT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/TCHOL_J.xpt) | [Documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/TCHOL_J.htm) |

| TRIGLY_J.xpt | [XPORT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/TRIGLY_J.xpt) | [Documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/TRIGLY_J.htm) |

| ALQ_J.xpt | [XPORT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/ALQ_J.xpt) | [Documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/ALQ_J.htm) |

| PAQ_J.xpt | [XPORT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/PAQ_J.xpt) | [Documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/PAQ_J.htm) |

| SLQ_J.xpt | [XPORT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SLQ_J.xpt) | [Documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SLQ_J.htm) |

| SMQ_J.xpt | [XPORT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SMQ_J.xpt) | [Documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SMQ_J.htm) |

| MCQ_J.xpt | [XPORT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/MCQ_J.xpt) | [Documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/MCQ_J.htm) |

The merger begins with DEMO_J and joins BMX_J, BPX_J, BPQ_J, DIQ_J, GLU_J, HDL_J, TCHOL_J, TRIGLY_J, ALQ_J, PAQ_J, SLQ_J, SMQ_J and MCQ300C from MCQ_J by SEQN using validated one-to-one left joins. All 9254 demographics records are retained before eligibility restriction. GLU_J and TRIGLY_J duplicate fasting weights are verified equal by design.py. The known pandas XPORT zero conversion artifact 5.397605346934028e-79 is restored to exact zero. No generic cutoff is applied.

prepare_raw.py writes data/prepared/NHANES_2017_2018_Merged.csv and outputs/raw_source_manifest.json, including SHA256 checksums and record counts. design.py reconstructs every predictor. data/reference/definite_predictor_benchmark.csv is optional and used only to verify inherited derivations, never as a model input.

The frozen split is authoritative in protocol/psu_assignment.csv. Do not delete or search a new split to improve held-out performance.