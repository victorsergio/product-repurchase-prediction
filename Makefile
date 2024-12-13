LOCAL_DIR = /Users/victor.penaloza/Documents/mldm-project
REMOTE_HOST = cluster
REMOTE_DIR = /home/pv10123z/mldm-project

# Sync specific folders and files from local to remote
SYNC_FOLDERS = $(addprefix $(LOCAL_DIR)/, src scripts requirements.txt)

# Sync specified folders and files
sync:
	@echo "Syncing folders and files: $(SYNC_FOLDERS)..."
	rsync -avz $(SYNC_FOLDERS) $(REMOTE_HOST):$(REMOTE_DIR)/

# Clean the remote directory (use with caution)
clean:
	ssh $(REMOTE_HOST) "rm -rf $(REMOTE_DIR)/*"


