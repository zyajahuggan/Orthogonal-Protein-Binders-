from pathlib import Path
from IPython import embed

project = "chai"
project_path = Path(__file__).resolve().parent.parent.parent / f"{project}" / "outputs" 
inputs_dir = Path(__file__).resolve().parent.parent / "inputs" / f"{project}" / f"{project}_cif_files.txt"
patterns_to_drop = ['6a', '6b', '13_xaaaa', '13_xaaab']
counter = 0
with open(inputs_dir, "w", encoding="utf-8") as file:
    for project in project_path.iterdir():
        for jobs in project.iterdir():
            job_name = jobs.stem
            if job_name == "metrics" :
                continue 
            if any(pattern in job_name.lower() for pattern in patterns_to_drop) and "dhd" in project.stem.lower():
                continue
            else:
                file.write(f"{jobs}/pred.model_idx_0.cif \n")
                counter +=1
        print(counter)
        counter = 0


                    
        

            
    

#embed()