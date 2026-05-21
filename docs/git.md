
git remote add sam3 org-16943930@github.com:facebookresearch/sam3.git
git subtree add --prefix third_party/sam3 sam3 main --squash
git remote add openpi git@github.com:Physical-Intelligence/openpi.git
git subtree add --prefix third_party/openpi openpi main --squash

