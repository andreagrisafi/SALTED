ENV_BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH=$ENV_BASE_DIR/bin:$ENV_BASE_DIR/soapfast/scripts:$ENV_BASE_DIR/soapfast:$ENV_BASE_DIR/soapfast/scripts/uncertainty:$PATH

unset ENV_BASE_DIR
