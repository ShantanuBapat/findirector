# infra/ecr.tf
#
# WHAT THIS CREATES:
#   An ECR (Elastic Container Registry) repository — the private AWS registry
#   where the FinDirector app container image is stored, so EKS nodes can pull it.
#   The image journey: built locally -> docker push to HERE -> EKS pulls from here.

resource "aws_ecr_repository" "app" {
  name = "${var.project_name}-app"

  # Scan images for known vulnerabilities on push (free, good hygiene).
  image_scanning_configuration {
    scan_on_push = true
  }

  # IMMUTABLE tags would block overwriting a tag; MUTABLE lets us re-push :latest
  # during development. (Production often prefers IMMUTABLE + unique tags.)
  image_tag_mutability = "MUTABLE"

  tags = {
    Name = "${var.project_name}-app"
  }
}

# --- Lifecycle policy: auto-expire old untagged images to control storage cost ---
# Without this, every push accumulates; old untagged layers pile up. This keeps
# only the most recent images and cleans up the rest.
resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images, expire older"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}