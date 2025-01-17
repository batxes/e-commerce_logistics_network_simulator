import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum
import simpy
import random

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
                random.uniform(48.0, 52.0),  # latitude
                random.uniform(8.0, 12.0)    # longitude
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
            
            # Record completion
            self.orders_processed.append({
                'order_id': order.id,
                'warehouse_id': warehouse.id,
                'processing_time': warehouse.processing_time,
                'delivery_time': delivery_time,
                'total_time': self.env.now - order.timestamp,
                'priority': order.priority.name
            })
    
    def find_optimal_warehouse(self, order: Order) -> Warehouse:
        """Finds the best warehouse to fulfill an order based on inventory and distance"""
        valid_warehouses = []
        
        for warehouse in self.warehouses:
            # Check inventory
            if not all(warehouse.inventory.get(item, 0) > 0 for item in order.items):
                continue
                
            # Calculate score based on distance and capacity
            distance = self.calculate_distance(
                warehouse.location,
                order.destination
            )
            capacity_score = warehouse.capacity / max(sum(warehouse.inventory.values()), 1)
            score = distance * (1 - capacity_score)
            
            valid_warehouses.append((warehouse, score))
        
        if not valid_warehouses:
            return None
            
        return min(valid_warehouses, key=lambda x: x[1])[0]
    
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
    
    return simulator.get_performance_metrics()

# Example usage with results display
if __name__ == "__main__":
    # Run the simulation
    print("Running logistics simulation for 1 week...")
    metrics = run_simulation()
    
    # Display results in a formatted way
    print("\nSimulation Results:")
    print("-" * 50)
    
    print(f"Total Orders: {metrics['total_orders']}")
    print(f"Successful Orders: {metrics['successful_orders']}")
    print(f"Failed Orders: {metrics['failed_orders']}")
    print(f"Success Rate: {metrics['success_rate']:.2%}")
    
    print("\nTiming Metrics (hours):")
    print(f"Average Processing Time: {metrics['avg_processing_time']:.2f}")
    print(f"Average Delivery Time: {metrics['avg_delivery_time']:.2f}")
    print(f"Average Total Time: {metrics['avg_total_time']:.2f}")
    
    print("\nOrders by Priority:")
    for priority, count in metrics['orders_by_priority'].items():
        print(f"{priority}: {count}")
