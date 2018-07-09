gcloud compute instances create dlnd-1 \
    --machine-type n1-standard-2 --zone us-east1-d \
    --accelerator type=nvidia-tesla-k80,count=1 \
    --image-family ubuntu-1604-lts --image-project ubuntu-os-cloud --boot-disk-size 50GB\
    --maintenance-policy TERMINATE --restart-on-failure \
    --metadata startup-script=''