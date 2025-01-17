
# e-commerce logistics simulator

Discrete event simulation of e-commerce logistics, modeling warehouse operations and deliveries across Germany with real-time visualization. Tech: Python, SimPy (discrete event simulator), Folium, Plotly, Pandas.

1. **Simulation Development**
- Implements a discrete event simulation using SimPy
- Models complex logistics network with multiple warehouses, inventory management, and delivery vehicles
- Handles different order priorities and realistic geographical distances

2. **System Architecture**
- Uses object-oriented design with clear separation of concerns
- Implements dataclasses for clean data structures
- Includes type hints for better code maintainability

3. **Business Logic & Optimization**
- Warehouse selection algorithm considers:
  - Current inventory levels
  - Distance to delivery location
  - Warehouse capacity
- Priority-based delivery time calculations
- Realistic order generation using Poisson process

4. **Analysis & Metrics**
- Comprehensive performance metrics including:
  - Success/failure rates
  - Processing times
  - Delivery times
  - Priority distribution

5. **Real-world Considerations**
- Geographical distances using Haversine formula
- Resource constraints (delivery vehicles)
- Variable processing times
- Multiple inventory items

