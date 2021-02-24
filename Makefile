.PHONY: ssh
ssh: ## Copy ssh keys to hosts
	scripts/setup-ssh.sh

.PHONY: install-mpi4py
install-mpi4py: ## Build mpi4py from source
	node/install-mpi4py.sh

.PHONY: send
send: ## Send the "scripts/" and "data/" direcotries to hosts
	scripts/send.sh

.PHONY: clean
clean: ## Clean build files and directories
	rm -rfd mpi4py*/
	rm -rf mpi4py*.tar.gz


.PHONY: help
help: ## Print this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {gsub("\\\\n",sprintf("\n%22c",""), $$2);printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
