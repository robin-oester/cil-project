echo "Running CIL Project code compliance check"

echo "Running auto-formatters"

conda run -n cil-project isort . > /dev/null
conda run -n cil-project autopep8 cil_project --recursive --in-place --pep8-passes 2000 > /dev/null
conda run -n cil-project black cil_project --verbose --config black.toml > /dev/null

echo "Running linters"

if conda run -n cil-project flake8 ; then
    echo "No flake8 errors"
else
    echo "flake8 errors"
    exit 1
fi

if conda run -n cil-project isort . --check --diff ; then
    echo "No isort errors"
else
    echo "isort errors"
    exit 1
fi

if conda run -n cil-project black --check cil_project --config black.toml ; then
    echo "No black errors"
else
    echo "black errors"
    exit 1
fi

if conda run -n cil-project pylint cil_project ; then
    echo "No pylint errors"
else
    echo "pylint errors"
    exit 1
fi

if conda run -n cil-project mypy cil_project ; then
    echo "No mypy errors"
else
    echo "mypy errors"
    exit 1
fi

echo "Successful code compliance check"
