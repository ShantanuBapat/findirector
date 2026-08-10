# infra/backend.tf
#
# WHAT THIS CONFIGURES:
#   Terraform's "backend" — WHERE it stores its state (the record of everything
#   it has created). We use the S3 bucket created during bootstrap, so state is
#   durable, recoverable (bucket versioning is on), and usable from any machine.
#
# backend "s3" settings:
#   bucket       : the state bucket (created by hand in the console).
#   key          : the path/filename of the state file WITHIN the bucket.
#   region       : the bucket's region.
#   encrypt      : encrypt the state object at rest in S3.
#   use_lockfile : S3-native state locking (Terraform 1.10+) — prevents two runs
#                  from corrupting state simultaneously, without needing a
#                  separate DynamoDB table.
#
# NOTE: backend config CANNOT use variables (var.*) — it's read before variables
#   are processed — so these values are written literally here.

terraform {
  backend "s3" {
    bucket       = "findirector-tfstate-svb"
    key          = "findirector/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
    profile      = "findirector"
  }
}