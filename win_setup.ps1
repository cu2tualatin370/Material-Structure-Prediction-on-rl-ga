Miniconda/Anaconda
$EnvName = "myproj"

if (-not (conda env list | Select-String $EnvName)) {
    conda env create -f environment.yml -n $EnvName
}

conda activate $EnvName
python main.py $args