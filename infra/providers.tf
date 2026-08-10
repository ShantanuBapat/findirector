# infra/providers.tf
#
# WHAT THIS CONFIGURES:
#   The AWS provider — how Terraform talks to AWS — plus default_tags that get
#   auto-applied to EVERY resource this project creates.
#
# provider "aws" block:
#   region       : which AWS region to operate in (us-east-1, matching setup).
#   profile      : which local AWS CLI profile / credentials to use (the
#                  Identity Center SSO profile you log in with).
#   default_tags : a tag set stamped onto every taggable resource automatically,
#                  so you never hand-tag and never forget one. This is what makes
#                  "filter cost by Project" and "find everything to tear down"
#                  reliable.
#
# The tag VALUES reference variables (var.*) defined in variables.tf, so names
# like the project and environment aren't hardcoded here.

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = var.owner
    }
  }
}