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
        self.active_deliveries = []
        
        # Increase number of vehicles
        self.delivery_vehicles = simpy.Resource(env, capacity=100)
        
        # Start processes
        self.env.process(self.generate_orders())
        
    def generate_orders(self):
        """Generates orders following a Poisson process"""
        order_id = 0
        while True:
            # Wait for next order
            yield self.env.timeout(random.expovariate(self.mean_order_rate))
            
            # Create order with coordinates within Germany
            items = self.generate_random_items()
            destination = (
                random.uniform(47.0, 54.0),  # latitude range for Germany
                random.uniform(6.0, 15.0)    # longitude range for Germany
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
        
        # Calculate delivery time before starting delivery
        delivery_time = self.calculate_delivery_time(
            warehouse.location,
            order.destination,
            order.priority
        )
        
        # Deliver order
        with self.delivery_vehicles.request() as vehicle:
            yield vehicle
            
            # Add to active deliveries
            delivery = {
                'order_id': order.id,
                'start_pos': warehouse.location,
                'end_pos': order.destination,
                'start_time': self.env.now,
                'end_time': self.env.now + delivery_time,
                'priority': order.priority
            }
            self.active_deliveries.append(delivery)
            
            # Wait for delivery time
            yield self.env.timeout(delivery_time)
            
            # Remove from active deliveries
            try:
                self.active_deliveries.remove(delivery)
            except ValueError:
                print(f"Warning: Could not remove delivery {order.id} from active deliveries")
            
            # Record completion
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
            # Check if warehouse has enough inventory for all items
            if not all(warehouse.inventory.get(item, 0) > 0 for item in order.items):
                print(f"Warehouse {warehouse.id} skipped: insufficient inventory")
                continue
                
            # Calculate inventory availability ratio
            total_inventory = sum(warehouse.inventory.values())
            
            # Skip warehouse if inventory is too low relative to capacity
            if total_inventory < len(order.items):
                print(f"Warehouse {warehouse.id} skipped: inventory too low ({total_inventory} items)")
                continue
                
            # Skip warehouse if inventory exceeds capacity
            if total_inventory > warehouse.capacity:
                print(f"Warehouse {warehouse.id} skipped: inventory ({total_inventory}) exceeds capacity ({warehouse.capacity})")
                continue
                
            # Calculate distance
            distance = self.calculate_distance(
                warehouse.location,
                order.destination
            )
            
            # Calculate capacity ratio (between 0 and 1)
            capacity_ratio = total_inventory / warehouse.capacity
            if capacity_ratio > 1:
                continue  # Skip if ratio is invalid
                
            # Calculate score (lower is better)
            # Balance between distance and inventory levels
            score = distance * (1 + (1 - capacity_ratio))
            
            valid_warehouses.append((warehouse, score))
        
        if not valid_warehouses:
            print(f"No valid warehouses found for order {order.id}")
            return None
            
        # Add debug print
        selected = min(valid_warehouses, key=lambda x: x[1])[0]
        print(f"\nOrder {order.id} destination: {order.destination}")
        print(f"Selected warehouse: {selected.id} (location: {selected.location})")
        print(f"Available warehouses and scores:")
        for w, score in valid_warehouses:
            total_inv = sum(w.inventory.values())
            print(f"  Warehouse {w.id}: score={score:.2f}, inventory={total_inv}/{w.capacity}")
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
        # Faster speed for more visible movement
        base_time = distance / 100  # 100 km/h average speed
        
        # Adjust for priority
        priority_multiplier = {
            OrderPriority.STANDARD: 0.5,
            OrderPriority.EXPRESS: 0.3,
            OrderPriority.SAME_DAY: 0.2
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

class LogisticsVisualizer:
    def __init__(self, simulator):
        self.simulator = simulator
        self.processed_orders_df = pd.DataFrame()  # Initialize empty
        self.failed_orders_df = pd.DataFrame()    # Initialize empty
        
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
        
        self.map = None
        self.delivery_layer = None
        self.vehicle_layer = None
        self.last_update = 0
        
    def create_map(self):
        """Creates an interactive map with real-time delivery visualization"""
        # Create base map centered on Germany
        self.map = folium.Map(location=[51.1657, 10.4515], zoom_start=6)
        
        # Create feature groups for different layers
        warehouse_layer = folium.FeatureGroup(name='Warehouses')
        self.delivery_layer = folium.FeatureGroup(name='Delivery Routes')
        self.vehicle_layer = folium.FeatureGroup(name='Completed Deliveries')
        
        # Add warehouses
        for _, warehouse in self.warehouses_df.iterrows():
            folium.CircleMarker(
                location=[warehouse['lat'], warehouse['lon']],
                radius=10,
                popup=warehouse['name'],
                color='red',
                fill=True
            ).add_to(warehouse_layer)
        
        # Add completed delivery routes
        processed_orders = pd.DataFrame(self.simulator.orders_processed)
        if not processed_orders.empty:
            for _, order in processed_orders.iterrows():
                # Find warehouse location
                warehouse = self.warehouses_df[self.warehouses_df['id'] == order['warehouse_id']].iloc[0]
                
                # Draw route from warehouse to destination
                route_coords = [
                    [warehouse['lat'], warehouse['lon']],
                    [order['dest_lat'], order['dest_lon']]
                ]
                
                # Color based on priority
                color_map = {
                    'STANDARD': 'blue',
                    'EXPRESS': 'orange',
                    'SAME_DAY': 'red'
                }
                
                folium.PolyLine(
                    route_coords,
                    color=color_map[order['priority']],
                    weight=2,
                    opacity=0.5,
                    popup=f"Order {order['order_id']} - {order['priority']}"
                ).add_to(self.delivery_layer)
                
                # Add destination marker
                folium.CircleMarker(
                    location=[order['dest_lat'], order['dest_lon']],
                    radius=3,
                    color=color_map[order['priority']],
                    fill=True,
                    popup=f"Destination - Order {order['order_id']}"
                ).add_to(self.delivery_layer)
        
        # Add all layers to map
        self.map.add_child(warehouse_layer)
        self.map.add_child(self.delivery_layer)
        self.map.add_child(self.vehicle_layer)
        
        # Add layer control
        folium.LayerControl().add_to(self.map)
        
        return self.map
    
    def update_visualization(self, current_time):
        """Updates delivery routes and vehicle positions"""
        if current_time - self.last_update < 0.5:  # Update every 30 minutes of sim time
            return
            
        self.last_update = current_time
        
        # Remove old layers from map
        for layer in [self.delivery_layer, self.vehicle_layer]:
            if layer and layer in self.map._children.values():
                for key in list(self.map._children.keys()):
                    if self.map._children[key] == layer:
                        del self.map._children[key]
        
        # Create new layers
        self.delivery_layer = folium.FeatureGroup(name='Delivery Routes')
        self.vehicle_layer = folium.FeatureGroup(name='Completed Deliveries')
        
        # Draw active deliveries
        for delivery in self.simulator.active_deliveries:
            try:
                # Calculate vehicle position
                progress = (current_time - delivery['start_time']) / (delivery['end_time'] - delivery['start_time'])
                progress = min(max(progress, 0), 1)  # Clamp between 0 and 1
                
                current_lat = delivery['start_pos'][0] + (delivery['end_pos'][0] - delivery['start_pos'][0]) * progress
                current_lon = delivery['start_pos'][1] + (delivery['end_pos'][1] - delivery['start_pos'][1]) * progress
                
                # Draw delivery route
                route_coords = [
                    [delivery['start_pos'][0], delivery['start_pos'][1]],
                    [delivery['end_pos'][0], delivery['end_pos'][1]]
                ]
                
                # Color based on priority
                color_map = {
                    OrderPriority.STANDARD: 'blue',
                    OrderPriority.EXPRESS: 'orange',
                    OrderPriority.SAME_DAY: 'red'
                }
                
                # Add route line
                folium.PolyLine(
                    route_coords,
                    color=color_map[delivery['priority']],
                    weight=4,
                    opacity=1.0,
                    popup=f"Order {delivery['order_id']}"
                ).add_to(self.delivery_layer)
                
                # Add vehicle marker
                folium.CircleMarker(
                    location=[current_lat, current_lon],
                    radius=5,
                    color='green',
                    fill=True,
                    fill_opacity=1.0,
                    popup=f"Vehicle - Order {delivery['order_id']}",
                    tooltip=f"Priority: {delivery['priority'].name}"
                ).add_to(self.vehicle_layer)
            except Exception as e:
                print(f"Error updating delivery {delivery.get('order_id', 'unknown')}: {str(e)}")
        
        # Add new layers to map
        self.map.add_child(self.delivery_layer)
        self.map.add_child(self.vehicle_layer)

    def plot_order_timeline(self):
        """Creates a timeline visualization of order processing"""
        # Update DataFrames with current data
        self.processed_orders_df = pd.DataFrame(self.simulator.orders_processed)
        self.failed_orders_df = pd.DataFrame(self.simulator.failed_orders)
        
        # Check if we have data to plot
        if len(self.processed_orders_df) == 0:
            print("No orders processed yet")
            return None

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Orders Over Time', 'Processing Times Distribution', 
                          'Orders by Priority', 'Delivery Times by Distance')
        )

        # Orders over time
        if 'timestamp' in self.processed_orders_df.columns:
            fig.add_trace(
                go.Histogram(
                    x=self.processed_orders_df['timestamp'],
                    nbinsx=20,
                    name='Processed Orders'
                ),
                row=1, col=1
            )

        # Processing times distribution
        if 'processing_time' in self.processed_orders_df.columns:
            fig.add_trace(
                go.Histogram(
                    x=self.processed_orders_df['processing_time'],
                    name='Processing Times'
                ),
                row=1, col=2
            )

        # Orders by priority
        if 'priority' in self.processed_orders_df.columns:
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
        if all(col in self.processed_orders_df.columns for col in ['delivery_time', 'total_time']):
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
        # Update DataFrames with current data
        self.processed_orders_df = pd.DataFrame(self.simulator.orders_processed)
        
        # Check if we have data to plot
        if len(self.processed_orders_df) == 0:
            print("No orders processed yet")
            return None

        # Group orders by warehouse
        if all(col in self.processed_orders_df.columns for col in ['warehouse_id', 'order_id', 'processing_time']):
            warehouse_orders = self.processed_orders_df.groupby('warehouse_id').agg({
                'order_id': 'count',
                'processing_time': 'mean',
            }).reset_index()

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Orders per Warehouse', 'Average Processing Times')
            )

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
        else:
            print("Missing required columns for warehouse utilization plot")
            return None

# Modified run_simulation function to collect visualization data
def run_simulation_with_viz(duration: int = 24*7, update_interval: float = 0.5):
    """Run simulation with visualization updates"""
    env = simpy.Environment()
    
    # Create warehouses
    warehouses = [
        Warehouse(
            id=1,
            location=(50.1109, 8.6821),  # Frankfurt
            capacity=1000,
            processing_time=1.0,
            inventory={item: 100 for item in ['A', 'B', 'C', 'D', 'E', 'F']}
        ),
        Warehouse(
            id=2,
            location=(48.1351, 11.5820),  # Munich
            capacity=800,
            processing_time=1.2,
            inventory={item: 80 for item in ['A', 'B', 'C', 'D', 'E', 'F']}
        ),
        Warehouse(
            id=3,
            location=(52.5200, 13.4050),  # Berlin
            capacity=720, 
            processing_time=0.8, 
            inventory={item: 120 for item in ['A', 'B', 'C', 'D', 'E', 'F']}
        )
    ]
    
    # Create simulator with higher order rate for more visible activity
    simulator = LogisticsSimulator(env, warehouses, mean_order_rate=5.0)  # 5 orders per hour
    viz = LogisticsVisualizer(simulator)
    
    # Run simulation
    env.run(until=duration)
    
    # Create final map with all completed routes
    map_viz = viz.create_map()
    
    # Add all completed delivery routes to the map
    delivery_layer = folium.FeatureGroup(name='Completed Deliveries')
    
    for order in simulator.orders_processed:
        # Get warehouse location
        warehouse = next(w for w in warehouses if w.id == order['warehouse_id'])
        
        # Create route coordinates
        route_coords = [
            [warehouse.location[0], warehouse.location[1]],  # Start at warehouse
            [order['dest_lat'], order['dest_lon']]          # End at destination
        ]
        
        # Color based on priority
        color_map = {
            'STANDARD': 'blue',
            'EXPRESS': 'orange',
            'SAME_DAY': 'red'
        }
        
        # Add the delivery route
        folium.PolyLine(
            route_coords,
            color=color_map[order['priority']],
            weight=3,
            opacity=0.8,
            popup=f"Order {order['order_id']} - {order['priority']}"
        ).add_to(delivery_layer)
        
        # Add destination marker
        folium.CircleMarker(
            location=[order['dest_lat'], order['dest_lon']],
            radius=4,
            color=color_map[order['priority']],
            fill=True,
            popup=f"Destination - Order {order['order_id']}"
        ).add_to(delivery_layer)
    
    # Add the delivery layer to the map
    map_viz.add_child(delivery_layer)
    
    # Add layer control
    folium.LayerControl().add_to(map_viz)
    
    # Create final plots
    timeline_plot = viz.plot_order_timeline()
    warehouse_plot = viz.plot_warehouse_utilization()
    
    return map_viz, timeline_plot, warehouse_plot

# Example usage
if __name__ == "__main__":
    network_map, timeline_plot, warehouse_plot = run_simulation_with_viz(
        #duration=24,  # Run for 1 day
        duration=48,  # Run for more days
        update_interval=0.1  # Update frequently
    )
    
    # Save visualizations
    print("Saving visualizations...")
    network_map.save("logistics_network_map.html")
    timeline_plot.write_html("order_timeline.html")
    warehouse_plot.write_html("warehouse_analysis.html")
    print("Done! Check the HTML files for the visualizations.")


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
