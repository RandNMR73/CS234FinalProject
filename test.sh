# python screen_agent.py --mode train --model-name dqn-test --screen-width 32 --screen-height 64 --buffer-size 10000 --total-timesteps 25000
# python screen_agent.py --mode train --model-name dqn-test --screen-width 64 --screen-height 128 --buffer-size 10000 --total-timesteps 25000
# python screen_agent.py --mode train --model-name dqn-test --screen-width 128 --screen-height 256 --buffer-size 10000 --total-timesteps 25000
python screen_agent.py --mode train --model-name dqn-test --screen-width 256 --screen-height 512 --buffer-size 10000 --total-timesteps 25000

# python screen_agent.py --mode train --model-name dqn-test --screen-width 32 --screen-height 64 --buffer-size 10000 --total-timesteps 25000 --num-screens 
# python screen_agent.py --mode train --model-name dqn-test --screen-width 32 --screen-height 64 --buffer-size 10000 --total-timesteps 25000
# python screen_agent.py --mode train --model-name dqn-test --screen-width 32 --screen-height 64 --buffer-size 10000 --total-timesteps 25000
# python screen_agent.py --mode train --model-name dqn-test --screen-width 32 --screen-height 64 --buffer-size 10000 --total-timesteps 25000

# python screen_agent.py --mode predict --model-name dqn-test5.zip --model-dir "output/train/05-06-2024-08-54/" --total-timesteps 10 --traj-log-freq 1