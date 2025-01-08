update Bank filter .name = <str>$bank_name set {
  products += (insert Product {
    name := <str>$name,
    interest := <float64>$interest,
    min_trust := <int64>$min_trust,
    end_date := <int64>$end_date
  })
}
