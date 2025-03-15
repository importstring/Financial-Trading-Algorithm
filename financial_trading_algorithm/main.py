#!/usr/bin/env python3

import asyncio
import logging
from pathlib import Path
from typing import Optional
import argparse
import sys
from rich.console import Console
from rich.logging import RichHandler

from utils.config_manager import ConfigManager
from api.openai_client import OpenAIClient
from api.perplexity_client import PerplexityClient
from api.ollama_client import OllamaClient
from trading.agent import Agent
from utils.api_key_manager import APIKeyManager

# Set up rich console for beautiful output
console = Console()

def setup_logging(config: ConfigManager):
    """Set up logging configuration."""
    log_level = config.get('Logging', 'level', fallback='INFO')
    log_format = config.get('Logging', 'format')
    log_file = config.get_path('Logging', 'file')
    
    # Create log directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            RichHandler(console=console, rich_tracebacks=True),
            logging.FileHandler(log_file)
        ]
    )

def setup_api_keys(config: ConfigManager) -> bool:
    """Set up API keys if not already configured."""
    api_key_manager = APIKeyManager()
    
    # Check for existing keys
    services = ['openai', 'perplexity']
    missing_keys = [s for s in services if not api_key_manager.get_key(s)]
    
    if not missing_keys:
        return True
    
    console.print("[yellow]Some API keys are missing. Let's set them up.[/yellow]")
    
    try:
        for service in missing_keys:
            key = console.input(f"Enter your {service} API key: ")
            if not api_key_manager.set_key(service, key):
                console.print(f"[red]Failed to save {service} API key[/red]")
                return False
        return True
    except Exception as e:
        console.print(f"[red]Error setting up API keys: {e}[/red]")
        return False

async def initialize_clients() -> tuple[OpenAIClient, PerplexityClient, OllamaClient]:
    """Initialize API clients."""
    try:
        openai_client = OpenAIClient()
        perplexity_client = PerplexityClient()
        ollama_client = OllamaClient()
        return openai_client, perplexity_client, ollama_client
    except Exception as e:
        console.print(f"[red]Error initializing API clients: {e}[/red]")
        raise

async def run_trading_system(
    config: ConfigManager,
    initial_balance: float = 100000.0,
    risk_tolerance: float = 0.02
) -> None:
    """Run the main trading system."""
    try:
        # Initialize trading agent
        agent = Agent(initial_balance=initial_balance, risk_tolerance=risk_tolerance)
        
        # Start trading process
        console.print("[green]Starting trading system...[/green]")
        await agent.begin()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")
    except Exception as e:
        console.print(f"[red]Error in trading system: {e}[/red]")
        raise
    finally:
        # Cleanup
        if 'agent' in locals():
            await agent.finalize_execution()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Financial Trading Algorithm")
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=100000.0,
        help="Initial trading balance"
    )
    parser.add_argument(
        "--risk-tolerance",
        type=float,
        default=0.02,
        help="Risk tolerance (0-1)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config file"
    )
    return parser.parse_args()

async def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Initialize configuration
        config = ConfigManager()
        
        # Set up logging
        setup_logging(config)
        
        # Set up API keys if needed
        if not setup_api_keys(config):
            console.print("[red]Failed to set up API keys. Exiting.[/red]")
            sys.exit(1)
        
        # Initialize API clients
        openai_client, perplexity_client, ollama_client = await initialize_clients()
        
        # Print startup information
        console.print("\n[bold green]Financial Trading Algorithm[/bold green]")
        console.print(f"Initial Balance: ${args.initial_balance:,.2f}")
        console.print(f"Risk Tolerance: {args.risk_tolerance:.1%}")
        console.print("API Clients: [green]✓[/green] Initialized")
        console.print("Configuration: [green]✓[/green] Loaded")
        console.print("Logging: [green]✓[/green] Configured\n")
        
        # Run trading system
        await run_trading_system(
            config,
            initial_balance=args.initial_balance,
            risk_tolerance=args.risk_tolerance
        )
        
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        logging.exception("Fatal error in main")
        sys.exit(1)
    finally:
        # Cleanup
        if 'openai_client' in locals():
            await openai_client.close()
        if 'perplexity_client' in locals():
            await perplexity_client.close()
        if 'ollama_client' in locals():
            await ollama_client.close()

if __name__ == "__main__":
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Program terminated by user[/yellow]")
        sys.exit(0) 