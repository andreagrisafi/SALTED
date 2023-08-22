ENV_BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SALTEDPATH=$ENV_BASE_DIR/src
export SALTEDLIBS=$ENV_BASE_DIR

unset ENV_BASE_DIR
