# Northwind Robotics — Knowledge Base (Test Fixture)

> Synthetic document for RAG ingestion and retrieval tests. Every name, number and
> policy here is fictional. Questions with expected answers are listed at the end.

## Company overview

Northwind Robotics GmbH was founded in March 2019 and is headquartered in Rotterdam,
Netherlands. The company builds autonomous material-handling robots for warehouses and
has 84 employees across offices in Rotterdam and Austin, Texas. The fiscal year ends on
31 March. Annual revenue for FY2025 was 41.2 million EUR.

Northwind sells directly in the EU and through the partner network in North America.
The Rotterdam lab is closed on the first Friday of every month for maintenance.

## Products

### Atlas-3 autonomous forklift

The Atlas-3 is the flagship warehouse robot. It carries a maximum payload of 1,200 kg,
reaches a top speed of 2.5 m/s, and uses a 48 V lithium iron phosphate battery that lasts
14 hours on a full charge. It ships with 16 lidar sensors and a 360-degree safety bumper.

Firmware 4.2.1 fixed the lidar drift issue that affected units built before June 2024.
Northwind recommends charging a new Atlas-3 for at least 6 hours before its first shift.

### Beacon-2 indoor positioning sensor

The Beacon-2 is a 2.4 GHz indoor positioning sensor with a range of 300 m and an accuracy
of ±10 cm. It is powered over Ethernet (PoE+) and must be mounted at least 2.5 m above
the floor. A typical aisle needs one Beacon-2 every 40 m.

### Fleet Manager

Fleet Manager is the optional cloud console for dispatching and monitoring robots. The
license covers up to 25 robots per site and includes firmware management, utilization
reports and alert routing.

## Support and service levels

Support is available 07:00–19:00 CET, Monday to Friday. Priority 1 incidents are handled
24/7. Contact support@northwind.example or call +31 10 555 0142.

| Severity | First response | Target resolution |
|----------|----------------|-------------------|
| P1 (line down) | 15 minutes | 4 hours |
| P2 (degraded) | 4 hours | 2 business days |
| P3 (question) | 2 business days | best effort |

A P1 incident is any condition that stops a production line. Opening a P1 requires a
phone call; email alone is not sufficient for P1.

## Warranty and returns

Robots carry a 24-month warranty from delivery. Batteries are covered for 12 months.
Returns require an RMA number, which support issues within 5 business days. Returns are
accepted within 30 days of delivery; after 30 days a 15% restocking fee applies.

## Deployment checklist

1. Survey the floor and mark all aisles and charging bays.
2. Mount Beacon-2 units at least 2.5 m above the floor.
3. Charge each Atlas-3 for at least 6 hours before the first shift.
4. Update every robot to firmware 4.2.1 or later.
5. Run the calibration route twice and confirm ±10 cm positioning.
6. Register the robot in Fleet Manager and assign a site.

## Security and compliance

Northwind holds SOC 2 Type II (2024) and ISO 27001 (2025) certifications. Customer data
is encrypted with AES-256 at rest and TLS 1.3 in transit. Fleet Manager supports SAML 2.0
single sign-on and role-based access control. An external penetration test is performed
annually, and all customer data is stored in the EU.

## List pricing

| Item | List price (EUR) |
|------|------------------|
| Atlas-3 | 48,000 |
| Beacon-2 | 640 |
| Fleet Manager | 12,000 per year |
| Onsite training | 2,400 per day |

Prices exclude VAT and shipping. Volume discounts start at 10 units.

## Troubleshooting

```text
E-104  lidar drift          -> update firmware to 4.2.1
E-221  battery temperature  -> let the robot cool for 30 minutes
E-330  beacon timeout       -> check PoE+ and the 2.5 m mounting height
```

If an error persists after these steps, collect the diagnostics bundle and open a P1
incident by phone.

## Glossary

- **AMR** — autonomous mobile robot.
- **PoE+** — Power over Ethernet, 802.3at.
- **RMA** — return merchandise authorization.
- **SLA** — service level agreement.

## Change log

- 2026-11-08 — v1.0 — initial test fixture.
