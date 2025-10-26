## Setup
``` uv install && source .venv/bin/activate```

Recommended command to run

```python3 main.py --model openai/gpt-oss-120b:exacto```

Add your API keys. 
Setting up the google one is super easy just go here follow the instructions. https://developers.google.com/custom-search/v1/overview
```
export OPENROUTER_API_KEY=""
export GOOGLE_API_KEY=""
```

## Usage Examples

Some of these may occasionally break. WIP. Markdown rendering is spotty. Latex rendering should work but doesn't.

Here's some examples of tool calling:

### Fetch -> "Summarize https://en.wikipedia.org/wiki/SpaceX_Starship"
```
🛠️  Model wants to run `fetch_url` with args {'url': 'https://en.wikipedia.org/wiki/SpaceX_Starship'}
🤖 **SpaceX Starship – Overview**

SpaceX’s Starship is a fully reusable, super‑heavy‑lift launch system designed to carry both crew and cargo to Earth orbit, the Moon, Mars, and
beyond. It consists of two main stages:
```

### Google Search -> "find the top 10 computer mice and give me a summary"
```
🤖 Here are the top‑10 computer mice based on recent search results, along with a brief summary of each:

                                                                                                                                               
  #    Mouse                         Key Features                               Typical Price (USD)   Why It Ranks High                        
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
  1    Logitech MX Master 3          Advanced ergonomic shape, MagSpeed         99‑119                Consistently praised for                 
                                     scroll wheel, Flow multi‑device                                  productivity‑focused design, long        
                                     switching, USB‑C charging                                        battery life, and precise tracking.      
  2    Razer DeathAdder V2           High‑precision 20,000 DPI sensor,          49‑69                 Popular among gamers for reliable        
                                     ergonomic right‑hand shape, Razer                                performance and affordable price.        
                                     Speedflex cable                                                                                           
  3    Microsoft Surface Mouse       Minimalist design, Bluetooth, 1‑year       49‑59                 Ideal for Windows users who want a       
                                     battery, smooth scrolling                                        sleek, low‑profile mouse for office      
                                                                                                      work.                                    
  4    Logitech G502 HERO            16,000 DPI HERO sensor, 11 programmable    79‑99                 Favoured by gamers for customization     
                                     buttons, adjustable weight system                                and high accuracy.                       
  5    Apple Magic Mouse 2           Multi‑touch surface, Bluetooth,            79‑99                 Preferred by macOS users for its         
                                     rechargeable, sleek aluminum body                                gesture support and minimalist look.     
  6    Corsair Dark Core RGB/SE      Wireless (2.4 GHz & Bluetooth), 18,000     69‑89                 Offers both wired and wireless           
                                     DPI sensor, customizable RGB, Qi                                 flexibility for gamers.                  
                                     wireless charging (SE)                                                                                    
  7    Logitech MX Anywhere 3        Compact, USB‑C, MagSpeed wheel, works on   69‑79                 Great for travelers needing a portable,  
                                     any surface, up to 70 days battery                               high‑performance mouse.                  
  8    SteelSeries Rival 3           8,500 DPI sensor, durable 60 M click       29‑39                 Budget‑friendly option that still        
                                     lifespan, RGB lighting                                           delivers solid gaming performance.       
  9    HP X4000b Bluetooth Mouse     Bluetooth, ergonomic shape, 2‑year         19‑24                 Very affordable, suitable for casual     
                                     battery, smooth scrolling                                        users and students.                      
  10   Logitech G Pro X Superlight   25,600 DPI HERO sensor, 63 g ultra‑light   149‑159               Top choice for esports pros who need     
                                     weight, wireless 2.4 GHz, PTFE feet                              the lightest, fastest mouse.             
                                                                                                                                               

Summary                                                                                                                                        

 • Productivity‑focused mice (Logitech MX Master 3, MX Anywhere 3, Microsoft Surface Mouse) dominate the list because of ergonomic designs,    
   multi‑device support, and long battery life.                                                                                                
 • Gaming mice (Razer DeathAdder V2, Logitech G502 HERO, Corsair Dark Core, SteelSeries Rival 3, Logitech G Pro X Superlight) are highlighted  
   for high‑DPI sensors, programmable buttons, and customizable weights or lighting.                                                           
 • Price range spans from under 20 for basic Bluetooth models up to150+ for high‑end esports devices, giving options for every budget.         
 • Connectivity trends show a mix of wired, wireless (2.4 GHz), and Bluetooth, with many newer models offering USB‑C charging.                 

These selections reflect current consumer reviews, expert recommendations, and sales popularity as of the latest web search. If you’d like     
deeper details on any specific mouse (e.g., full specs, pros/cons, or purchase links), just let me know!   
```

### Python execution -> "calculate how much money I'd have with $100k invested in the S&p 500 at historical returns for 30 years"

```
🛠️  Model wants to run `python_execute` with args {'code': "principal = 100000\nrate = 0.10\nyears = 30\nfv = principal * 
((1+rate)**years)\nprint(f'Future value: ${fv:,.2f}')"}
🤖 The future value of a $100,000 investment in the S&P 500, assuming a historical average annual return of **10 %**, would be:
Future Value ≈ $1,744,940.23 after 30 years. 
```