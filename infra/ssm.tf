# infra/ssm.tf
#
# WHAT THIS CREATES:
#   The SSM access path — how you reach the private RDS from your laptop without
#   opening any inbound ports or using SSH keys. Pieces: an IAM role (permissions
#   the instance wears), the EC2 tunnel instance itself, and VPC endpoints (so the
#   private instance can reach the SSM service without going to the internet).
#
# Built in dependency order: IAM role -> instance profile -> instance -> endpoints.

# --- IAM role: the permissions the SSM instance wears ---
# Trust policy: only the EC2 service may assume this role (so an EC2 instance can
# wear it). Without permissions ON the instance, it can't talk to SSM.
resource "aws_iam_role" "ssm_instance" {
  name = "${var.project_name}-ssm-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
    }]
  })

  tags = {
    Name = "${var.project_name}-ssm-instance-role"
  }
}

# --- Attach the AWS-managed SSM policy: WHAT the role can do ---
# AmazonSSMManagedInstanceCore grants exactly the permissions an instance needs to
# be managed by SSM (register with the service, receive sessions) — nothing more.
resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.ssm_instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# --- Instance profile: the wrapper that lets an EC2 instance wear the role ---
# An IAM role can't be attached to an EC2 instance directly; it's attached via an
# "instance profile" (a thin container around the role). This is an AWS quirk.
resource "aws_iam_instance_profile" "ssm_instance" {
  name = "${var.project_name}-ssm-instance-profile"
  role = aws_iam_role.ssm_instance.name
}

# --- Look up the latest Amazon Linux 2023 AMI (has the SSM agent preinstalled) ---
# A data source READS from AWS rather than creating anything. We query for the
# newest AL2023 image owned by Amazon, so we don't hardcode an AMI ID (they differ
# by region and change over time).
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# --- The SSM tunnel instance: a tiny private EC2 you port-forward through ---
# Sits in a private subnet with NO public IP. Wears the SSM instance profile (so
# SSM can manage it) and the db_client security group (so it may reach RDS:5432).
# You never SSH to it — you reach it via SSM Session Manager.
resource "aws_instance" "ssm_tunnel" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = "t3.micro"
  subnet_id              = module.vpc.private_subnets[0]
  iam_instance_profile   = aws_iam_instance_profile.ssm_instance.name
  vpc_security_group_ids = [aws_security_group.db_client.id]

  associate_public_ip_address = false

  tags = {
    Name = "${var.project_name}-ssm-tunnel"
  }
}

# --- Security group for the VPC endpoints: allow HTTPS from within the VPC ---
# The endpoints are reached over HTTPS (443). This SG permits 443 inbound from
# anything in the VPC's CIDR, so the SSM instance can talk to the endpoints.
resource "aws_security_group" "vpc_endpoints" {
  name        = "${var.project_name}-vpc-endpoints"
  description = "Allow HTTPS from within the VPC to the interface endpoints"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description = "HTTPS from within the VPC"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-vpc-endpoints"
  }
}

# --- The three interface VPC endpoints SSM needs ---
# Private doorways from the VPC directly to the SSM services, so SSM traffic never
# goes over the internet. All three are required for Session Manager to work.
# for_each creates one endpoint per service name, keeping this DRY.
resource "aws_vpc_endpoint" "ssm" {
  for_each = toset(["ssm", "ssmmessages", "ec2messages"])

  vpc_id              = module.vpc.vpc_id
  service_name        = "com.amazonaws.${var.aws_region}.${each.key}"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = module.vpc.private_subnets
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true

  tags = {
    Name = "${var.project_name}-vpce-${each.key}"
  }
}