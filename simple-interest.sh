#!/bin/bash

# Simple Interest Calculation Script

echo "Enter Principal Amount:"
read P

echo "Enter Rate of Interest (per annum):"
read R

echo "Enter Time Period (in years):"
read T

# Calculate Simple Interest (SI = P * R * T / 100)
SI=$(echo "scale=2; ($P * $R * $T) / 100" | bc)

echo "Simple Interest: $SI"
