// Document Setup
#let title = "CS 148 Homework 1 Responses"
#let author = "Gavin Hua"

// Heading numbering (1.a.i)
// #set heading(numbering: "1.a.i")

// Page setup with header
#set page(
  numbering: "1",
  number-align: right,
  header: [
    #smallcaps([#title])
    #h(1fr) #smallcaps([#author])
    #line(length: 100%)
    #v(-10pt)
    #line(length:100%)
  ]
)

// Text formatting
#set par(justify: true)
#set text(
  font: "TeX Gyre Pagella",
  size: 11pt,
)

= 2.4
If the corpus is large enough, then we might have the following merge sequence:
`d-o`, `do-g`, `dog-!` and `dog-?`, which will render `dog!` and `dog?` two completely distinct tokens. 
This shows that the naive merging method's flaw where it disregards linguistic boundaries.

= 2.5
Longest token: `_accomplishment` (with a leading space, typeset as `_`).
This makes sense because TinyStories is a dataset of simple stories, so we expect common english words (like accomplishment) to get merged into single tokens.

Training completed in 295 seconds, with peak RAM 10.6 GB.

= 2.7
== (a)
Approximately 4.1 tokens.

== (b)
We have 10k tokens. The integer datatypes go `uint8/16/32`, and the smallest one with $2^k > 10000$ is $k = 16$ (with $2^16 = 65536$).


= 3.2.4
=== Softmax
Subtract the maximum element from all $v_i$s before softmaxing.
Then, the largest exponent has value $0$.
This is equivalent to dividing the numerator and denominator by $exp(v_max)$ beforehand.
=== Masking
Hadamard product with the boolean mask matrix.

= 4.1
Perplexity might be better because it represents the "average number of tokens" the LLM is choosing from as if it were choosing randomly at each step, which is more interpretable.


= 6
The ratio of two exponential elements in the softmax denominator is given by $exp(v_i / tau) / exp(v_j / tau) = exp((v_i - v_j) / tau)$.
If $tau -> 0$, this ratio will diverge to $oo$ or converge to $0$, depending on the sign of $v_i - v_j$.
This means that the largest element in ${v_i}$ will dominate the softmax, and all probability will be concentrated on that element. 
On the other hand, if $tau -> oo$, then the ratio will converge to $1$, which means all elements are weighted equally.

= 7.2
== Train
Validation loss is calculated by sampling $20$ random batches from the validation set and computing the average loss on them. 
#figure(
  image("baseline.png"),
  caption: "Baseline learning curves"
)
== Generate
We seed the text generation with the `<|endoftext|>` token.

$tau = 0.0, p=1.0$: this parameter set gives decently fluent output but suffers from some repetition (`"Thank you, Mom!"` is repeated twice).
```
<|endoftext|>
Once upon a time, there was a little boy named Tim. Tim had a toy car that he loved to play with. One day, Tim's toy car got stuck in a tree. Tim was sad and didn't know what to do.
Tim's mom saw him and asked, "What's wrong, Tim?" Tim said, "My car is stuck in the tree!" His mom had an idea. She took Tim to the tree and put a long stick against the tree. Tim was happy and said, "Thank you, Mom!"
Tim's mom helped him get the car down from the tree. Tim was so happy and said, "Thank you, Mom!" He played with his toy car all day long. And from that day on, Tim always remembered to be careful when playing with his toys.
<|endoftext|>
```

+ Factor 1: increasing $tau$
  - $tau = 0.5$ (newline output by model): there is arguably more "creativity" in this story but it's less logical compared to the $tau = 0$ story. 
  ```
  <|endoftext|>

  Once upon a time, there was a little girl who loved to explore. She would go to the park and look around. One day, she saw a big box in the park and she was curious, so she went to take a look. She opened the box and found a lot of toys inside. She was so excited!
  The little girl wanted to play with all the toys, but she was too excited to pick out the best toys she could find. She started to pick up all the toys and put them in a big box. Then she grabbed the box and opened it.
  Inside the box was a big surprise! It was a new toy that was just for her. She was so happy and excited. She played with the new toys all day and had lots of fun.
  <|endoftext|>
  ```
  - $tau = 10.0$: output is completely unintelligible.
  ```
  <|endoftext|> Comet pump snugg plot Em til wraps Pink gave magnets Terryizzyict airpl FrBr lips unwrappedide meal whiteetter Suzie rockedAlwaysddy Bub cl sideun hoop oak beans measured saus leanedalkJonbert Rudy chased growls climb tiptoedAm he crack Trimet sailedummies learns rabbit on atelawlish snack grab Blink hornsracadabraiding spell nice regrettedues grateful to zip bak wheelsrs goaluriousStill geeselor flowing tricked Ashley cuddly room radio games wake future sk bar center sheet mindful Beef Light pres PepperMark freedom Bert crow windingartled hist germ tricksiss cur bent appro lifted Ro shoneatch Dolly slippPenny almost teles damagednderction good Granny chip daisies hugeever child kings screwd sounded recognize� imagination� loudest curly brickack intelligent pointsath pre declirrelfaFinny becomes ladyb high attitude demwe paying bu tumbled fought Joke spells picksank agreement letter present holes norm birdc isatheres webs yummy donkey poop cane spread heel pumpkin yellow messagesabies adm tickets glittering usedrel explor smallestedic flapped zipperda locadiumbubul bedtime thanking col gir bathrobe talking hangHe hid caredcast Linda prun jellyfish handsome dark impatientrophone believeents ur suits reason villag combed vide needle dance pecksacked shelfumbling pumpkins unl loadedreat mayor praying peck<|endoftext|>
  ```

+ Decreasing $p$ (with $tau = 0.5$ fixed)
  - $p = 0.50$: the output is coherent and similar in quality to $p = 1.0$ at the same temperature. Trimming the bottom half of the probability mass seems to have little visible effect here, likely because at $tau = 0.5$ the distribution is already fairly peaked and the tail tokens rarely get sampled anyway.
  ```
  <|endoftext|>
  Once upon a time, there was a little boy named Tim. Tim had a toy car that he loved to play with. One day, Tim's toy car got stuck in a tree. Tim was sad and didn't know what to do.
  Tim's mom saw him and asked, "What's wrong, Tim?" Tim told her about his toy car. His mom said, "Don't worry, we can get your toy car back." They went to the tree and found the toy car stuck in the tree. Tim was happy again.
  Tim and his mom went home and played with the toy car. They had lots of fun. Tim learned that even if something unexpected happens, it can still be fun. The moral of the story is to always be kind and help others.
  <|endoftext|>
  ```
  - $p = 0.01$: the output is nearly identical to the $tau = 0$ (greedy) case. This makes sense because $V(p)$ with $p = 0.01$ almost always contains only the single most probable token, so we are effectively doing greedy decoding regardless of temperature.
  ```
  <|endoftext|>
  Once upon a time, there was a little boy named Tim. Tim had a toy car that he loved to play with. One day, Tim's toy car got stuck in a tree. Tim was sad and didn't know what to do.
  Tim's mom saw him and asked, "What's wrong, Tim?" Tim said, "My car is stuck in the tree!" His mom had an idea. She took Tim to the tree and put a long stick against the tree. Tim was happy and said, "Thank you, Mom!"
  Tim's mom helped him get the car down from the tree. Tim was so happy and said, "Thank you, Mom!" He played with his toy car all day long. And from that day on, Tim always remembered to be careful when playing with his toys.
  <|endoftext|>
  ```


= 7.3
== Ablation 1
With the previous best-performing hyperparameters, the model training is unstable, and we observe loss spikes and immediate plateauing. 
I cut the training short at around 2000 iterations because the loss was not decreasing after 1000 iterations. 
#figure(
  image("no_ln.png"),
  caption: "Pink: No LayerNorm, trained with optimal hyperparameters. Blue: baseline"
)

After removing LayerNorm, I lowered the learning rate by a factor of $10$.
While stability recovered, the final loss plateaued at a higher level than before.
#figure(
  image("no_ln_lower_lr.png"),
  caption: "Gray: No LayerNorm, trained with lowered learning rate. Blue: baseline"
)
- The reason for destabilization is likely that LayerNorm acts as a per-token re-centering and re-scaling before each sublayer, thus keeping the input to the MHSA and FFN in a consistent range.
  Without it, small perturbations upstream will be amplified, and there will be no guarantee that MHSA/FFN's gradients will be kept reasonable. 
- The reason for lower learning rate recovering performance is that even though the gradients are poorly scaled without LayerNorm, the optimizer isn't overshooting as aggressively, so training stays in the region where the loss surface is still smooth. 


== Ablation 2
#figure(
  image("no_sin_pe.png"),
  caption: "Blue: baseline. Yellow: NoPE, trained with optimal hyperparameters"
)