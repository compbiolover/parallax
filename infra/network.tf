# Public subnets, and no NAT gateway. That is a deliberate cost decision worth
# stating plainly: a NAT gateway is ~$32/month before a byte moves through it —
# a quarter of the entire running budget — and its only job here would be to
# hide tasks that accept no inbound traffic in the first place.
#
# What replaces it is a security group with **zero ingress rules**. The task has
# a public IP and can reach the internet; nothing on the internet can open a
# connection to it.
#
# The corollary is the thing that bites people: with no public IP *and* no NAT,
# a Fargate task cannot reach ECR, and the failure is "CannotPullContainerError"
# with no mention of networking. The setting that prevents it is
# `AssignPublicIp = "ENABLED"` in the state machine's NetworkConfiguration
# (scheduler.tf) — not anything in this file. `map_public_ip_on_launch` below
# governs instances in the subnet and does *not* cover a Fargate task, which
# gets its address from the RunTask call.

resource "aws_vpc" "main" {
  cidr_block           = "10.20.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = var.name }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = var.name }
}

# Two AZs because Fargate capacity is per-AZ: a single subnet turns one AZ's bad
# afternoon into a failed run.
resource "aws_subnet" "public" {
  count = 2

  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = { Name = "${var.name}-public-${count.index}" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = { Name = "${var.name}-public" }
}

resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# Egress only. No ingress rule exists, and none should: the tasks fetch feeds,
# call Anthropic, and send SMTP. Nothing ever connects *to* them.
resource "aws_security_group" "tasks" {
  name        = "${var.name}-tasks"
  description = "Parallax Fargate tasks: outbound only, no inbound whatsoever"
  vpc_id      = aws_vpc.main.id

  tags = { Name = "${var.name}-tasks" }
}

resource "aws_vpc_security_group_egress_rule" "all" {
  security_group_id = aws_security_group.tasks.id
  description       = "Feeds, GDELT, Anthropic, SMTP on 587, S3, ECR"
  ip_protocol       = "-1"
  cidr_ipv4         = "0.0.0.0/0"
}
