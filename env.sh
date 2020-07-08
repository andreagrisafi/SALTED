ENV_BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RHOMLPATH=$ENV_BASE_DIR/src

unset ENV_BASE_DIR
