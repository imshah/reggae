# Employee Onboarding Process

## Overview
The onboarding process is triggered when HR marks a candidate as "Hired" in the ATS.
An automated webhook then creates accounts and schedules orientation.

## Steps
1. HR sets status to Hired (trigger).
2. IdentityService provisions SSO and email.
3. Manager assigns a buddy and first-week plan.
4. Orientation is scheduled for day 1.

## Ownership
HR owns the trigger; the Platform team owns IdentityService. No owner is named
for the buddy assignment step.
