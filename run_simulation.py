# Create a new file called run_simulation.py

from e_commerce_logistics_network_simulator_viz import (
    run_simulation_with_viz,
    OrderPriority,
    Warehouse
)

def main():
    # Run a shorter simulation with more frequent orders
    print("Starting simulation...")
    
    # Run simulation for 24 hours with higher order rate
    network_map, timeline_plot, warehouse_plot = run_simulation_with_viz(
        duration=24,  # 1 day
        update_interval=0.1  # Update more frequently
    )
    
    print("Simulation complete. Saving visualizations...")
    
    # Save all visualizations
    network_map.save("logistics_network_map.html")
    timeline_plot.write_html("order_timeline.html")
    warehouse_plot.write_html("warehouse_analysis.html")
    
    print("Visualizations saved:")
    print("- logistics_network_map.html")
    print("- order_timeline.html")
    print("- warehouse_analysis.html")

if __name__ == "__main__":
    main()