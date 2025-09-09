package models

import "time"

type Product struct {
	ID          uint
	SellerID    uint
	CategoryID  uint
	Title       string
	Description string
	Brand       string
	Status      int
	Score       int
	CreatedAt   time.Time
	UpdatedAt   time.Time
}
