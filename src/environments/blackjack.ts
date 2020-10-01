import { sum } from "../util";
import { Environment, Observation, Sample } from "./core";

type Ace = 1;
type Jack = 10;
type Queen = 10;
type King = 10;
type Card = Ace | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Jack | Queen | King;

type Deck = readonly Card[];
type Hand = readonly Card[];

const unshuffledDeck = Array.from({ length: 52 }, (_, i) =>
	Math.min(10, (i % 13) + 1),
) as Deck;

const hasUsableAce = (hand: Hand): boolean =>
	hand.includes(1) && sum(hand) + 10 <= 21;

const sumHand = (hand: Hand): number =>
	hasUsableAce(hand) ? sum(hand) + 10 : sum(hand);

const isBust = (hand: Hand): boolean => sumHand(hand) > 21;

const scoreHand = (hand: Hand): number => (isBust(hand) ? 0 : sumHand(hand));

const compareHands = (player: Hand, dealer: Hand): number => {
	const playerScore = scoreHand(player);
	const dealerScore = scoreHand(dealer);
	return Number(playerScore > dealerScore) - Number(playerScore < dealerScore);
};

const isNatural = (hand: Hand): boolean =>
	hand.length === 2 && hand.includes(1) && hand.includes(10);

const fisherYates = <T>(array: readonly T[]): readonly T[] => {
	const copy = [...array];
	for (let i = copy.length - 1; i > 0; --i) {
		const j = Math.floor(Math.random() * (i + 1));
		[copy[i], copy[j]] = [copy[j], copy[i]];
	}
	return copy;
};

const shuffleDeck = (deck: Deck = unshuffledDeck): Deck => {
	return fisherYates(deck);
};

export class Blackjack implements Environment {
	public readonly name: string;
	public readonly numObservationDimensions: number;
	public readonly numActions: number;
	public readonly natural: boolean;

	private done: boolean;
	private deck: Deck;
	private dealer: Hand;
	private player: Hand;

	public constructor(natural = false) {
		this.name = "Blackjack";
		this.numObservationDimensions = 3;
		this.numActions = 2;
		this.natural = natural;

		this.done = true;
		this.deck = [];
		this.dealer = [];
		this.player = [];
	}

	private get state(): Observation {
		return [
			sumHand(this.player),
			this.dealer[0],
			Number(hasUsableAce(this.player)),
		];
	}

	private drawCard(): Card {
		if (this.deck.length < 1) {
			throw new Error("Not enough cards");
		}
		const card = this.deck[0];
		this.deck = this.deck.slice(1);
		return card;
	}

	private drawHand(): readonly Card[] {
		return [this.drawCard(), this.drawCard()];
	}

	private applyNaturalFactor(reward: number, hand: Hand): number {
		return reward === 1 && this.natural && isNatural(hand) ? 1.5 : reward;
	}

	public reset(): Observation {
		this.deck = shuffleDeck();
		this.player = this.drawHand();
		this.dealer = this.drawHand();
		this.done = false;
		return this.state;
	}

	public step(action: number): Sample {
		if (action >= this.numActions) {
			throw new Error("Action is not in range");
		}
		if (this.done) {
			throw new Error("Env is done");
		}

		// hit
		if (action) {
			this.player = [...this.player, this.drawCard()];
			return isBust(this.player)
				? {
						observation: this.state,
						reward: -1,
						done: true,
				  }
				: {
						observation: this.state,
						reward: 0,
						done: false,
				  };
		}

		// stick
		while (sumHand(this.dealer) < 17) {
			this.dealer = [...this.dealer, this.drawCard()];
		}

		const reward = compareHands(this.player, this.dealer);

		return {
			observation: this.state,
			reward: this.applyNaturalFactor(reward, this.player),
			done: true,
		};
	}
}
