import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum
import simpy
import random
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


# Definition of order priorities
# This defines 3 types of delivery priorities that customer can choose from.
class OrderPriority(Enum):
    STANDARD = 1
    EXPRESS = 2
    SAME_DAY = 3

@dataclass
class Warehouse:
    'Warehouse stores information about each warehouse (location, capacity, etc.)'
    id: int
    location: Tuple[float, float]  # lat, long
    capacity: int
    processing_time: float  # hours
    inventory: Dict[str, int]

@dataclass
class Order:
    'Order represents a customer order with items, delivery location, and priority'
    id: int
    items: List[str]
    destination: Tuple[float, float]
    priority: OrderPriority
    timestamp: float

class LogisticsSimulator:
    def __init__(self, env, warehouses: List[Warehouse], mean_order_rate: float):
        self.env = env
        self.warehouses = warehouses
        self.mean_order_rate = mean_order_rate
        self.orders_processed = []
        self.failed_orders = []
        
        # Resources
        self.delivery_vehicles = simpy.Resource(env, capacity=50)
        
        # Start processes
        self.env.process(self.generate_orders())
        
    def generate_orders(self):
        """Generates orders following a Poisson process"""
        """Generates random items
            Assigns random delivery location
            Sets priority (70% Standard, 20% Express, 10% Same-day)
        """
        order_id = 0
        while True:
            # Wait for next order
            yield self.env.timeout(random.expovariate(self.mean_order_rate))
            
            # Create order
            items = self.generate_random_items()
            destination = (
                random.uniform(47.0, 54.0),  # latitude
                random.uniform(6.0, 15.0)    # longitude
            )
            priority = random.choices(
                list(OrderPriority),
                weights=[0.7, 0.2, 0.1]
            )[0]
            
            order = Order(
                id=order_id,
                items=items,
                destination=destination,
                priority=priority,
                timestamp=self.env.now
            )
            
            # Process order
            self.env.process(self.process_order(order))
            order_id += 1
    
    def process_order(self, order: Order):
        """Processes a single order through the logistics network"""
        """
        This method simulates the entire order journey:

            Find best warehouse
            Reserve inventory
            Process order at warehouse
            Assign delivery vehicle
            Calculate and simulate delivery time
            Record completion
        """
        # Find best warehouse
        warehouse = self.find_optimal_warehouse(order)
        if not warehouse:
            self.failed_orders.append({
                'order_id': order.id,
                'reason': 'No warehouse with inventory',
                'timestamp': self.env.now
            })
            return
        
        # Reserve inventory
        for item in order.items:
            warehouse.inventory[item] -= 1
        
        # Process at warehouse
        yield self.env.timeout(warehouse.processing_time)
        
        # Deliver order
        with self.delivery_vehicles.request() as vehicle:
            yield vehicle
            
            # Calculate delivery time based on distance and priority
            delivery_time = self.calculate_delivery_time(
                warehouse.location,
                order.destination,
                order.priority
            )
            yield self.env.timeout(delivery_time)
            
            # Record completion with timestamp and destination coordinates
            self.orders_processed.append({
                'order_id': order.id,
                'warehouse_id': warehouse.id,
                'processing_time': warehouse.processing_time,
                'delivery_time': delivery_time,
                'total_time': self.env.now - order.timestamp,
                'priority': order.priority.name,
                'dest_lat': order.destination[0],
                'dest_lon': order.destination[1],
                'timestamp': self.env.now
            })
    
    def find_optimal_warehouse(self, order: Order) -> Warehouse:
        """Finds the best warehouse to fulfill an order based on inventory and distance"""
        valid_warehouses = []
        
        for warehouse in self.warehouses:
            # Check inventory
            if not all(warehouse.inventory.get(item, 0) > 0 for item in order.items):
                continue
                
            # Calculate distance
            distance = self.calculate_distance(
                warehouse.location,
                order.destination
            )
            
            # Calculate inventory availability ratio
            total_inventory = sum(warehouse.inventory.values())
            capacity_ratio = total_inventory / warehouse.capacity
            
            # Calculate score (lower is better)
            # Balance between distance and inventory levels
            score = distance * (1 + (1 - capacity_ratio))
            
            valid_warehouses.append((warehouse, score))
        
        if not valid_warehouses:
            return None
            
        # Add debug print
        selected = min(valid_warehouses, key=lambda x: x[1])[0]
        print(f"Order destination: {order.destination}")
        print(f"Selected warehouse: {selected.id} (location: {selected.location})")
        print(f"Available warehouses and scores:")
        for w, score in valid_warehouses:
            print(f"  Warehouse {w.id}: score={score:.2f}, inventory={sum(w.inventory.values())}")
        print("---")
        
        return selected
    
    @staticmethod
    def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculates distance between two lat/long points using Haversine formula"""
        lat1, lon1 = point1
        lat2, lon2 = point2
        R = 6371  # Earth's radius in kilometers
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2) * np.sin(dlat/2) +
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
             np.sin(dlon/2) * np.sin(dlon/2))
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c
    
    def calculate_delivery_time(self, 
                              warehouse_loc: Tuple[float, float],
                              destination: Tuple[float, float],
                              priority: OrderPriority) -> float:
        """Calculates delivery time based on distance and priority"""
        distance = self.calculate_distance(warehouse_loc, destination)
        base_time = distance / 50  # Assuming 50 km/h average speed
        
        # Adjust for priority
        priority_multiplier = {
            OrderPriority.STANDARD: 1.0,
            OrderPriority.EXPRESS: 0.7,
            OrderPriority.SAME_DAY: 0.5
        }
        
        return base_time * priority_multiplier[priority]
    
    def generate_random_items(self) -> List[str]:
        """Generates a random list of items for an order"""
        items = ['A', 'B', 'C', 'D', 'E', 'F']
        num_items = random.randint(1, 3)
        return random.choices(items, k=num_items)
    
    def get_performance_metrics(self) -> Dict:
        """Calculates key performance metrics from the simulation"""
        df = pd.DataFrame(self.orders_processed)
        failed_df = pd.DataFrame(self.failed_orders)
        
        metrics = {
            'total_orders': len(df) + len(failed_df),
            'successful_orders': len(df),
            'failed_orders': len(failed_df),
            'success_rate': len(df) / (len(df) + len(failed_df)),
            'avg_processing_time': df['processing_time'].mean(),
            'avg_delivery_time': df['delivery_time'].mean(),
            'avg_total_time': df['total_time'].mean(),
            'orders_by_priority': df['priority'].value_counts().to_dict()
        }
        
        return metrics
def run_simulation(duration: int = 24*7):  # 1 week simulation
    # Initialize simulation environment
    env = simpy.Environment()
    
    # Create warehouses
    warehouses = [
        Warehouse(
            id=1,
            location=(50.1, 8.7),  # Frankfurt
            capacity=1000,
            processing_time=1.0,
            inventory={item: 100 for item in ['A', 'B', 'C', 'D', 'E', 'F']}
        ),
        Warehouse(
            id=2,
            location=(48.1, 11.6),  # Munich
            capacity=800,
            processing_time=1.2,
            inventory={item: 80 for item in ['A', 'B', 'C', 'D', 'E', 'F']}
        ),
        Warehouse(
            id=3,
            location=(52.5, 13.4),  # Berlin
            capacity=1200,
            processing_time=0.8,
            inventory={item: 120 for item in ['A', 'B', 'C', 'D', 'E', 'F']}
        )
    ]
    
    # Create simulator
    simulator = LogisticsSimulator(env, warehouses, mean_order_rate=1.0)  # 1 order per hour on average
    
    # Run simulation
    env.run(until=duration)
    
    return simulator.get_performance_metrics(), simulator

class LogisticsVisualizer:
    def __init__(self, simulator):
        self.simulator = simulator
        self.processed_orders_df = pd.DataFrame(simulator.orders_processed)
        self.failed_orders_df = pd.DataFrame(simulator.failed_orders)
        
        # Convert warehouse data to DataFrame
        self.warehouses_df = pd.DataFrame([
            {
                'id': w.id,
                'lat': w.location[0],
                'lon': w.location[1],
                'capacity': w.capacity,
                'name': f'Warehouse {w.id}'
            }
            for w in simulator.warehouses
        ])

    def create_map(self):
        """Creates an interactive map showing warehouses and delivery paths"""
        # Create base map centered on Germany
        m = folium.Map(location=[51.1657, 10.4515], zoom_start=6)

        # Add warehouses
        for _, warehouse in self.warehouses_df.iterrows():
            folium.CircleMarker(
                location=[warehouse['lat'], warehouse['lon']],
                radius=10,
                popup=warehouse['name'],
                color='red',
                fill=True
            ).add_to(m)

        # Add heatmap of delivery destinations
        if not self.processed_orders_df.empty:
            # You would need to add destination coordinates to processed_orders during simulation
            heat_data = self.processed_orders_df[['dest_lat', 'dest_lon']].values.tolist()
            plugins.HeatMap(heat_data).add_to(m)

        return m

    def plot_order_timeline(self):
        """Creates a timeline visualization of order processing"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Orders Over Time', 'Processing Times Distribution', 
                          'Orders by Priority', 'Delivery Times by Distance')
        )

        # Orders over time
        fig.add_trace(
            go.Histogram(
                x=self.processed_orders_df['timestamp'],
                nbinsx=20,
                name='Processed Orders'
            ),
            row=1, col=1
        )

        # Processing times distribution
        fig.add_trace(
            go.Histogram(
                x=self.processed_orders_df['processing_time'],
                name='Processing Times'
            ),
            row=1, col=2
        )

        # Orders by priority
        priority_counts = self.processed_orders_df['priority'].value_counts()
        fig.add_trace(
            go.Bar(
                x=priority_counts.index,
                y=priority_counts.values,
                name='Order Priorities'
            ),
            row=2, col=1
        )

        # Delivery times vs distance
        fig.add_trace(
            go.Scatter(
                x=self.processed_orders_df['delivery_time'],
                y=self.processed_orders_df['total_time'],
                mode='markers',
                name='Delivery Time vs Total Time'
            ),
            row=2, col=2
        )

        fig.update_layout(height=800, width=1000, title_text="Logistics Network Analysis")
        return fig

    def plot_warehouse_utilization(self):
        """Creates a visualization of warehouse utilization over time"""
        # Group orders by warehouse
        warehouse_orders = self.processed_orders_df.groupby('warehouse_id').agg({
            'order_id': 'count',
            'processing_time': 'mean',
            'delivery_time': 'mean'
        }).reset_index()

        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Orders per Warehouse', 
                                         'Average Processing Times'))

        # Orders per warehouse
        fig.add_trace(
            go.Bar(
                x=warehouse_orders['warehouse_id'],
                y=warehouse_orders['order_id'],
                name='Total Orders'
            ),
            row=1, col=1
        )

        # Average processing times
        fig.add_trace(
            go.Bar(
                x=warehouse_orders['warehouse_id'],
                y=warehouse_orders['processing_time'],
                name='Avg Processing Time'
            ),
            row=1, col=2
        )

        fig.update_layout(height=400, width=800, 
                         title_text="Warehouse Performance Analysis")
        return fig

# Modified run_simulation function to collect visualization data
def run_simulation_with_viz():
    # Run the original simulation
    metrics, simulator = run_simulation()
    
    # Create visualizer
    viz = LogisticsVisualizer(simulator)
    
    # Generate plots
    network_map = viz.create_map()
    timeline_plot = viz.plot_order_timeline()
    warehouse_plot = viz.plot_warehouse_utilization()
    
    return network_map, timeline_plot, warehouse_plot

# Example usage
if __name__ == "__main__":
    network_map, timeline_plot, warehouse_plot = run_simulation_with_viz()
    
    # Save plots
    network_map.save("logistics_network_map.html")
    timeline_plot.write_html("order_timeline.html")
    warehouse_plot.write_html("warehouse_analysis.html")


## Example usage with results display
#if __name__ == "__main__":
#    # Run the simulation
#    print("Running logistics simulation for 1 week...")
#    metrics = run_simulation()
#    
#    # Display results in a formatted way
#    print("\nSimulation Results:")
#    print("-" * 50)
#    
#    print(f"Total Orders: {metrics['total_orders']}")
#    print(f"Successful Orders: {metrics['successful_orders']}")
#    print(f"Failed Orders: {metrics['failed_orders']}")
#    print(f"Success Rate: {metrics['success_rate']:.2%}")
#    
#    print("\nTiming Metrics (hours):")
#    print(f"Average Processing Time: {metrics['avg_processing_time']:.2f}")
#    print(f"Average Delivery Time: {metrics['avg_delivery_time']:.2f}")
#    print(f"Average Total Time: {metrics['avg_total_time']:.2f}")
#    
#    print("\nOrders by Priority:")
#    for priority, count in metrics['orders_by_priority'].items():
#        print(f"{priority}: {count}")
