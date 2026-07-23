# dockerHDDM 1.1 runtime

## Runtime contract

- Image tag: `hcp4715/hddm:1.1.0`
- Python 3.12
- NumPy 2.x
- ArviZ 1.1
- maintained PyMC2, Kabuki, HDDM, and ssm-simulators forks
- amd64 and arm64 images

## Pull and run

Docker Hub:

```bash
docker pull hcp4715/hddm:1.1.0
docker run -it --rm \
  -v "$(pwd):/home/jovyan/work" \
  -p 8888:8888 \
  hcp4715/hddm:1.1.0 \
  jupyter notebook
```

CNB mirror:

```bash
docker pull docker.cnb.cool/dockerhddm_sync/dockerhddm/hcp4715-hddm:1.1.0
docker tag docker.cnb.cool/dockerhddm_sync/dockerhddm/hcp4715-hddm:1.1.0 \
  hcp4715/hddm:1.1.0
```

PowerShell supports `$(pwd)`; `cmd.exe` needs an explicit host path. Quote paths
that contain spaces. Mount the project root, not an anonymous temporary folder,
because `--rm` deletes the container after exit.

## Smoke check

Inside the container:

```bash
python -c "import numpy, arviz, pymc, kabuki, hddm, ssms; \
print(numpy.__version__, arviz.__version__, pymc.__version__, hddm.__version__)"
python /path/to/dockerhddm-skill/scripts/inspect_environment.py
```

Confirm `arviz.loo_subsample`, multi-chain `sample`, and DataTree conversion
before using v1.1-only features.

## Persistence rules

- Host-mounted project: durable.
- Unmounted container filesystem: disposable.
- `models/*.db`: PyMC2 trace database.
- `models/*.hddm`: pickled HDDM model.
- `models/*.nc`: ArviZ/xarray inference artifact.
- Record image tag and submodule commits in the project README or run log.
